#include "detail/gemini.hpp"
#include "test_support.hpp"

#include <cail/cail.hpp>

#include <atomic>
#include <chrono>
#include <future>
#include <optional>
#include <utility>

namespace test {

struct PendingHttp {
  std::vector<cail::HttpRequest> requests;
  cail::HttpCompletion complete;
  std::stop_token stop;
  int blocking_calls{};

  void reply(const std::string& body, int status = 200) {
    auto callback = std::exchange(complete, {});
    check(static_cast<bool>(callback), "HTTP completion is pending");
    if (callback) {
      callback(cail::HttpResponse{.status_code = status,
                                  .headers = {{.name = "x-request-id", .value = "request-1"}},
                                  .body = body});
    }
  }
};

class DeferredTransport final : public cail::HttpTransport {
public:
  explicit DeferredTransport(std::shared_ptr<PendingHttp> state) : state_(std::move(state)) {}
  cail::Result<cail::HttpResponse> send(const cail::HttpRequest&, std::stop_token) override {
    ++state_->blocking_calls;
    return std::unexpected(cail::Error{.code = cail::ErrorCode::transport});
  }
  void send_async(cail::HttpRequest request, cail::HttpCompletion complete,
                  std::stop_token stop) override {
    state_->requests.push_back(std::move(request));
    state_->complete = std::move(complete);
    state_->stop = stop;
  }
  cail::Result<cail::HttpResponse> stream(const cail::HttpRequest& request,
                                          const cail::HttpDataHandler&,
                                          std::stop_token stop) override {
    return send(request, stop);
  }

private:
  std::shared_ptr<PendingHttp> state_;
};

void test_adapters() {
  const std::vector<std::string> responses{
      R"({"status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]}]})",
      R"({"stop_reason":"end_turn","content":[{"type":"text","text":"done"}]})",
      R"({"candidates":[{"content":{"parts":[{"text":"done"}]},"finishReason":"STOP"}]})",
      R"({"choices":[{"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}]})",
  };
  for (int adapter = 0; adapter < 8; ++adapter) {
    auto state = std::make_shared<PendingHttp>();
    auto transport = std::make_unique<DeferredTransport>(state);
    cail::LanguageModel model;
    int format = adapter;
    if (adapter == 0) {
      model = cail::create_openai({.api_key = "key"})("test", std::move(transport));
    } else if (adapter == 1) {
      model = cail::create_anthropic({.api_key = "key"})("test", std::move(transport));
    } else if (adapter == 2) {
      model = cail::create_gemini({.api_key = "key"})("test", std::move(transport));
    } else if (adapter == 3) {
      model = cail::create_local()("test", std::move(transport));
    } else {
      const std::vector<cail::OpenCodeApiFamily> families{
          cail::OpenCodeApiFamily::responses, cail::OpenCodeApiFamily::anthropic_messages,
          cail::OpenCodeApiFamily::gemini, cail::OpenCodeApiFamily::chat_completions};
      format = adapter - 4;
      model =
          cail::create_opencode({.api_key = "key"})("test", families[format], std::move(transport));
    }
    check(model.adapter_capabilities().async_generation,
          "every API family reports async generation");
    std::optional<cail::Result<cail::GenerationResponse>> result;
    std::vector<std::size_t> steps;
    int callbacks = 0;
    std::stop_source stop;
    const auto started = cail::generate_text_async(
        {.model = model,
         .system = "system",
         .prompt = "prompt",
         .tool_loop = {.stop = stop.get_token()},
         .session_id = "session",
         .max_output_tokens = 100,
         .middleware = {{.before_request =
                             [](cail::HttpRequest& request, const cail::MiddlewareContext&) {
                               request.headers.push_back({.name = "custom", .value = "yes"});
                             },
                         .after_response =
                             [&](const cail::HttpResponse& response,
                                 const cail::MiddlewareContext&) {
                               check(response.status_code == 200, "async response middleware runs");
                             },
                         .after_step =
                             [&](const cail::Result<cail::GenerationResponse>& response,
                                 const cail::MiddlewareContext& context) {
                               check(response.has_value(), "async step middleware receives result");
                               steps.push_back(context.step);
                             }}}},
        [&](cail::Result<cail::GenerationResponse> value) {
          ++callbacks;
          result = std::move(value);
        });
    check(started && !result && state->blocking_calls == 0 && state->requests.size() == 1,
          "async generation returns with a deferred HTTP request");
    check(state->stop == stop.get_token() &&
              state->requests.front().body.find("prompt") != std::string::npos &&
              state->requests.front().headers.back().name == "custom",
          "async adapters preserve cancellation, prompts, and request middleware");
    model = {};
    state->reply(responses[format]);
    check(result && *result && (*result)->text == "done" && callbacks == 1 &&
              steps == std::vector<std::size_t>{0},
          "async request owns the model until its one completion");
  }
}

void test_tool_loop() {
  std::vector<cail::LanguageModel::GenerationCompletion> pending;
  std::vector<cail::GenerationRequest> requests;
  std::stop_source stop;
  std::promise<void> follow_up;
  auto follow_up_ready = follow_up.get_future();
  cail::LanguageModel model(
      {}, {}, {},
      [&](cail::GenerationRequest request, cail::LanguageModel::GenerationCompletion complete,
          const std::stop_token& token) -> cail::Result<void> {
        check(token == stop.get_token(), "async tool requests receive stop token");
        requests.push_back(std::move(request));
        pending.push_back(std::move(complete));
        if (requests.back().step == 1) {
          follow_up.set_value();
        }
        return {};
      });
  int tool_calls = 0;
  auto tool = cail::tool<ToolInput, ToolOutput>(
      "count", "Count", [&](const ToolInput& input, const cail::ToolContext& context) {
        ++tool_calls;
        check(context.stop.stop_possible(), "tool handlers receive cancellation token");
        return ToolOutput{.count = static_cast<int>(input.query.size())};
      });
  std::promise<cail::Result<cail::GenerationResponse>> finished;
  auto final_result = finished.get_future();
  std::optional<cail::Result<cail::GenerationResponse>> result;
  const auto started =
      cail::generate_text_async({.model = model,
                                 .prompt = "count abc",
                                 .tools = {tool},
                                 .tool_loop = {.stop = stop.get_token()},
                                 .session_id = "session",
                                 .max_output_tokens = 123,
                                 .provider_options = {{"custom", "value"}}},
                                [&](auto response) { finished.set_value(std::move(response)); });
  check(started && !result && pending.size() == 1, "async tool loop starts first request");
  auto first = std::move(pending[0]);
  first(cail::GenerationResponse{
      .tool_calls = {{.id = "call-1", .name = "count", .arguments = R"({"query":"abc"})"}}});
  follow_up_ready.get();
  check(!result && tool_calls == 1 && pending.size() == 2 && requests[1].step == 1 &&
            requests[1].session_id == "session" && requests[1].max_output_tokens == 123 &&
            requests[1].provider_options.contains("custom") &&
            requests[1].messages.back().tool_call_id == "call-1",
        "async tool loop preserves options and submits tool output");
  auto second = std::move(pending[1]);
  second(cail::GenerationResponse{.text = "done"});
  result = final_result.get();
  check(result && *result && (*result)->turn.size() == 3 && (*result)->tool_results.size() == 1 &&
            (*result)->tool_results.front().output == R"({"count":3})",
        "async tool loop returns final response, history, and typed results");

  for (bool async : {false, true}) {
    std::stop_source cancelled;
    int executed = 0;
    auto cancel_tool = cail::tool<ToolInput, ToolOutput>(
        "cancel", "Cancel", [&](const ToolInput&, const cail::ToolContext& context) {
          check(context.stop.stop_possible(), "both tool loops propagate cancellation to handlers");
          ++executed;
          cancelled.request_stop();
          return ToolOutput{};
        });
    cail::GenerationResponse response{
        .tool_calls = {{.id = "one", .name = "cancel", .arguments = R"({"query":"x"})"},
                       {.id = "two", .name = "cancel", .arguments = R"({"query":"y"})"}}};
    auto cancelling_model = cail::LanguageModel(
        [response](const cail::GenerationRequest&, const std::stop_token&) { return response; }, {},
        {},
        [response](const cail::GenerationRequest&,
                   const cail::LanguageModel::GenerationCompletion& complete,
                   const std::stop_token&) -> cail::Result<void> {
          complete(response);
          return {};
        });
    cail::GenerateTextOptions options{.model = cancelling_model,
                                      .prompt = "cancel",
                                      .tools = {cancel_tool},
                                      .tool_loop = {.stop = cancelled.get_token()}};
    result.reset();
    if (async) {
      std::promise<cail::Result<cail::GenerationResponse>> completed;
      auto response = completed.get_future();
      check(cail::generate_text_async(options,
                                      [&](auto value) { completed.set_value(std::move(value)); })
                .has_value(),
            "inline async cancellation request starts");
      result = response.get();
    } else {
      result = cail::generate_text(options);
    }
    check(result && !*result && result->error().code == cail::ErrorCode::cancelled && executed == 1,
          "cancellation prevents remaining tools from executing in both loops");
    check(result && !*result && result->error().partial_response &&
              result->error().partial_response->tool_results.size() == 1 &&
              result->error().partial_response->turn.size() == 2,
          "both loops retain the tool result produced before cancellation");
  }
}

void test_inline_and_failures() {
  int rounds = 0;
  int callbacks = 0;
  auto tool = cail::tool<ToolInput, ToolOutput>("count", "Count",
                                                [](const ToolInput&) { return ToolOutput{}; });
  cail::LanguageModel model(
      {}, {}, {},
      [&](const cail::GenerationRequest& request,
          const cail::LanguageModel::GenerationCompletion& complete,
          const std::stop_token&) -> cail::Result<void> {
        ++rounds;
        if (request.step < 2000) {
          complete(cail::GenerationResponse{
              .tool_calls = {{.id = "call", .name = "count", .arguments = R"({"query":"x"})"}},
              .continuation_token = "continuation"});
        } else {
          complete(cail::GenerationResponse{.text = "done"});
        }
        return {};
      });
  std::promise<void> completed;
  auto done = completed.get_future();
  const auto inline_started = cail::generate_text_async(
      {.model = model, .prompt = "hello", .tools = {tool}, .tool_loop = {.max_rounds = 2000}},
      [&](auto response) {
        ++callbacks;
        check(response && response->tool_results.size() == 2000,
              "inline async completions finish a long tool loop");
        completed.set_value();
      });
  done.get();
  check(inline_started && callbacks == 1 && rounds == 2001,
        "inline callbacks do not recursively grow the stack");
  int rejected_callbacks = 0;
  const auto rejected =
      cail::generate_text_async({.model = model}, [&](auto) { ++rejected_callbacks; });
  std::stop_source stop;
  stop.request_stop();
  const auto cancelled = cail::generate_text_async(
      {.model = model, .prompt = "hello", .tool_loop = {.stop = stop.get_token()}},
      [&](auto) { ++rejected_callbacks; });
  check(!rejected && !cancelled && rejected_callbacks == 0,
        "validation and prior cancellation return an error without invoking completion");
  check(!cail::generate_text_async({.model = model, .prompt = "hello"}, {}),
        "empty completions fail");
  int requests = 0;
  cail::LanguageModel failing(
      {}, {}, {},
      [&](const cail::GenerationRequest&, const cail::LanguageModel::GenerationCompletion& complete,
          const std::stop_token&) -> cail::Result<void> {
        if (++requests == 2) {
          return std::unexpected(cail::Error{.code = cail::ErrorCode::transport});
        }
        complete(cail::GenerationResponse{
            .tool_calls = {{.id = "call", .name = "count", .arguments = R"({"query":"x"})"}}});
        return {};
      });
  std::promise<cail::Result<cail::GenerationResponse>> failure;
  auto failed = failure.get_future();
  const auto started =
      cail::generate_text_async({.model = failing, .prompt = "hello", .tools = {tool}},
                                [&](auto value) { failure.set_value(std::move(value)); });
  const auto result = failed.get();
  check(started && !result && result.error().code == cail::ErrorCode::transport,
        "follow-up initiation errors reach the completion");
  check(!result && result.error().partial_response &&
            result.error().partial_response->steps.size() == 1 &&
            result.error().partial_response->tool_results.size() == 1,
        "async follow-up failures preserve completed model and tool work");
}

void test_stream_failure_preserves_progress() {
  std::stop_source stop;
  std::optional<cail::Result<cail::GenerationResponse>> received;
  auto delivery = std::make_shared<cail::detail::StreamDelivery>(
      [](const cail::StreamEvent&) {}, [&](auto result) { received = std::move(result); },
      cail::AsyncOptions{}, stop.get_token());
  auto partial = std::make_shared<cail::GenerationResponse>();
  partial->tool_results.push_back({.call_id = "call", .name = "write", .output = "done"});
  stop.request_stop();
  delivery->finish(std::unexpected(
      cail::Error{.code = cail::ErrorCode::cancelled, .partial_response = partial}));
  check(received && !*received && received->error().partial_response == partial,
        "async stream delivery retains partial progress on cancellation");
}

void test_embeddings() {
  auto state = std::make_shared<PendingHttp>();
  auto model =
      cail::detail::make_embedding_model({.endpoint = "https://test/embeddings",
                                          .model = "test",
                                          .transport = std::make_unique<DeferredTransport>(state)});
  check(model.supports_async(), "embedding adapter reports async support");
  std::stop_source stop;
  std::optional<cail::Result<cail::Embedding>> result;
  const auto started =
      model.embed_async("hello", [&](auto value) { result = std::move(value); }, stop.get_token());
  check(started && !result && state->blocking_calls == 0 && state->stop == stop.get_token(),
        "async embeddings use nonblocking transport and forward cancellation");
  model = {};
  state->reply(R"({"model":"test","data":[{"index":0,"embedding":[0.2,0.4]}]})");
  check(result && *result && (*result)->dimensions == 2,
        "async single embedding survives model destruction");

  model =
      cail::detail::make_embedding_model({.endpoint = "https://test/embeddings",
                                          .model = "test",
                                          .transport = std::make_unique<DeferredTransport>(state)});
  std::optional<cail::Result<cail::EmbeddingBatch>> batch;
  check(model.embed_many_async({"one", "two"}, [&](auto value) { batch = std::move(value); })
            .has_value(),
        "async batch starts");
  state->reply(
      R"({"model":"test","data":[{"index":1,"embedding":[0.4]},{"index":0,"embedding":[0.2]}]})");
  check(batch && *batch && (*batch)->embeddings[0].values[0] == 0.2F,
        "async batches preserve input order");
  batch.reset();
  check(model
            .embed_many_async(
                {"one"}, [&](auto value) { batch = std::move(value); }, stop.get_token())
            .has_value(),
        "cancellable embedding starts");
  stop.request_stop();
  state->reply(R"({"model":"test","data":[{"index":0,"embedding":[0.2]}]})");
  check(batch && !*batch && batch->error().code == cail::ErrorCode::cancelled,
        "in-flight embeddings report cancellation");
  int callbacks = 0;
  check(!model.embed_many_async({}, [&](auto) { ++callbacks; }) &&
            !model.embed_async("", [&](auto) { ++callbacks; }) &&
            !model.embed_async("one", cail::EmbeddingModel::EmbeddingCompletion{}) &&
            !model.embed_many({"one"}, stop.get_token()) && callbacks == 0,
        "embedding validation and prior cancellation prevent requests");
  auto gemini_state = std::make_shared<PendingHttp>();
  cail::detail::gemini::EmbeddingClient gemini({.api_key = "key", .model = "test"}, {},
                                               std::make_unique<DeferredTransport>(gemini_state));
  batch.reset();
  check(gemini.embed_many_async({"one"}, [&](auto value) { batch = std::move(value); }).has_value(),
        "Gemini async embedding starts");
  gemini_state->reply(R"({"embeddings":[{"values":[0.2,0.4]}]})");
  check(batch && *batch && (*batch)->dimensions == 2, "Gemini decodes async embeddings");
}

void test_objects_and_agents() {
  auto state = std::make_shared<PendingHttp>();
  auto model = cail::create_local()("test", std::make_unique<DeferredTransport>(state));
  std::optional<cail::Result<Address>> object;
  const auto started = cail::generate_object_async<Address>(
      {.model = model, .prompt = "Name a city"}, [&](auto value) { object = std::move(value); });
  check(started && !object && state->requests.back().body.find("json_schema") != std::string::npos,
        "async object generation sends a typed schema");
  state->reply(
      R"({"choices":[{"message":{"content":"{\"city\":\"Paris\"}"},"finish_reason":"stop"}]})");
  check(object && *object && (*object)->city == "Paris",
        "async object generation decodes the C++ type");
  object.reset();
  check(cail::generate_object_async<Address>({.model = model, .prompt = "Name a city"},
                                             [&](auto value) { object = std::move(value); })
            .has_value(),
        "async incomplete object request starts");
  state->reply(R"({"choices":[{"message":{"content":"{}"},"finish_reason":"length"}]})");
  check(object && !*object && object->error().code == cail::ErrorCode::incomplete_response,
        "async objects preserve incomplete-response errors");
  check(!cail::generate_object_async<Address>({.model = model, .prompt = "hello"}, {}),
        "async objects reject an empty completion");

  auto memory = std::make_shared<InlineMemory>();
  check(
      memory
          ->append("conversation", {cail::Message{.content = {cail::TextPart{.text = "earlier"}}}})
          .has_value(),
      "agent history is seeded");
  std::optional<cail::Result<cail::GenerationResponse>> response;
  {
    cail::Agent agent({.model = model,
                       .instructions = "system",
                       .memory = memory,
                       .conversation_id = "conversation"});
    check(
        agent.generate_async("hello", [&](auto value) { response = std::move(value); }).has_value(),
        "async agent prompt starts with memory");
  }
  check(!response && state->requests.back().body.find("earlier") != std::string::npos &&
            state->requests.back().body.find("system") != std::string::npos,
        "async agents send instructions and conversation history");
  state->reply(R"({"choices":[{"message":{"content":"done"},"finish_reason":"stop"}]})");
  const auto history = memory->load("conversation");
  check(response && *response && history && history->size() == 3,
        "async agent survives destruction and stores its turn before completing");
  cail::Agent agent({.model = model, .memory = memory, .conversation_id = "conversation"});
  response.reset();
  check(agent
            .generate_async(
                cail::GenerationRequest{
                    .messages = {cail::Message{.content = {cail::TextPart{.text = "manual"}}}}},
                [&](auto value) { response = std::move(value); })
            .has_value(),
        "async agent accepts manually managed requests");
  state->reply(R"({"choices":[{"message":{"content":"done"},"finish_reason":"stop"}]})");
  check(response && *response && memory->load("conversation")->size() == 3,
        "async explicit agent requests bypass memory");
  response.reset();
  check(
      agent.generate_async("failure", [&](auto value) { response = std::move(value); }).has_value(),
      "failed async agent request starts");
  state->reply(R"({"error":{"message":"invalid request"}})", 400);
  check(response && !*response && response->error().http_status == 400 &&
            response->error().request_id == "request-1" &&
            memory->load("conversation")->size() == 3,
        "failed async requests preserve HTTP context and do not append memory");
}

void test_async_multimodal_memory() {
  auto state = std::make_shared<PendingHttp>();
  auto memory = std::make_shared<InlineMemory>();
  std::optional<cail::Result<cail::GenerationResponse>> response;
  const std::string binary{"\0\xff\x89", 3};
  {
    cail::Agent agent({.model = cail::create_openai({.api_key = "key"})(
                           "test", std::make_unique<DeferredTransport>(state)),
                       .memory = memory,
                       .conversation_id = "attachments"});
    check(agent
              .generate_async(
                  cail::Message{
                      .content = {cail::TextPart{.text = "Explain"},
                                  cail::ImagePart{.bytes = binary, .mime_type = "image/png"},
                                  cail::PdfPart{.bytes = "%PDF-1.7\n" + binary,
                                                .filename = "report.pdf"}}},
                  [&](auto value) { response = std::move(value); })
              .has_value(),
          "deferred multimodal agent starts");
  }
  check(!response && memory->load("attachments")->empty() &&
            state->requests.back().body.find("input_file") != std::string::npos &&
            state->requests.back().body.find("input_image") != std::string::npos,
        "deferred request owns its attachments without storing an unfinished turn");
  state->reply(
      R"({"status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]}]})");
  const auto stored = memory->load("attachments");
  check(response && *response && stored && stored->size() == 2 &&
            std::get<cail::ImagePart>(stored->front().content[1]).bytes == binary &&
            std::get<cail::PdfPart>(stored->front().content[2]).bytes == "%PDF-1.7\n" + binary,
        "attachments survive async agent destruction and persist after success");
}

} // namespace test

int main() {
  test::test_stream_failure_preserves_progress();
  test::test_adapters();
  test::test_tool_loop();
  test::test_inline_and_failures();
  test::test_embeddings();
  test::test_objects_and_agents();
  test::test_async_multimodal_memory();
  return test::failures == 0 ? 0 : 1;
}
