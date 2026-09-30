#include <cail/cail.hpp>

#include <atomic>
#include <chrono>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

struct Answer {
  std::string answer;
};

struct CountInput {
  std::string query;
};

namespace {

std::atomic<int> failures{};

void check(bool condition, std::string_view name) {
  if (!condition) {
    std::cerr << "FAIL: " << name << '\n';
    ++failures;
  }
}

cail::GenerationRequest prompt() {
  return {.messages = {cail::Message{
              .role = cail::MessageRole::user,
              .content = {cail::TextPart{.text = "Hello"}},
          }}};
}

cail::Task<void> coroutine_http_generation(cail::LanguageModel model) {
  const auto thread = std::this_thread::get_id();
  const auto response = co_await cail::generate_text_async({.model = model, .prompt = "Hello"});
  check(response && response->text == "Hello" && std::this_thread::get_id() == thread,
        "coroutine generation awaits live HTTP and resumes on its caller executor");
}

cail::Task<void> coroutine_http_stream(cail::LanguageModel model) {
  const auto thread = std::this_thread::get_id();
  std::string text;
  auto on_event = [&](const cail::StreamEvent& event) {
    check(std::this_thread::get_id() == thread,
          "live coroutine SSE events run on the caller executor");
    if (const auto* delta = std::get_if<cail::TextDelta>(&event))
      text += delta->text;
  };
  const auto streamed =
      co_await cail::stream_text_async({.model = model, .prompt = "Hello"}, on_event);
  check(streamed && text == "Hi",
        "coroutine streaming delivers live events before its final response");
}

cail::Task<void> await_tool_loop(cail::Task<cail::Result<cail::GenerationResponse>> task) {
  const auto tools = co_await std::move(task);
  check(tools && tools->text == "Done" && tools->tool_results.size() == 1,
        "coroutine generation awaits the whole live tool loop");
}

cail::Task<void> coroutine_agent_call(cail::Agent& agent, std::string prompt_text) {
  const auto response = co_await agent.generate_async(std::move(prompt_text));
  check(response.has_value(), "sequential coroutine agent calls await HTTP and memory persistence");
}

void coroutine_http(const std::string& base) {
  auto model = cail::create_local({.endpoint = base + "/chat"})("test-model");
  cail::run(coroutine_http_generation(model));
  cail::run(coroutine_http_stream(model));
  auto count =
      cail::tool<CountInput, int>("count", "Count characters", [](const CountInput& input) {
        return static_cast<int>(input.query.size());
      });
  auto tool_model = cail::create_local({.endpoint = base + "/tool-chat"})("test-model");
  cail::run(await_tool_loop(
      cail::generate_text_async({.model = tool_model, .prompt = "Count abc", .tools = {count}})));
  auto memory = std::make_shared<cail::InMemoryConversationMemory>();
  cail::Agent agent({.model = model, .memory = memory, .conversation_id = "coroutine"});
  cail::run(coroutine_agent_call(agent, "Hello"));
  cail::run(coroutine_agent_call(agent, "Again"));
  const auto history = memory->load("coroutine");
  check(history && history->size() == 4,
        "coroutine agent calls persist both completed conversation turns");
}

} // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    return 2;
  }
  const std::string base = argv[1];
  cail::detail::chat_completions::Client chat({.endpoint = base + "/chat", .api_key = "test-key"},
                                              "test-model");
  const auto chat_result = chat.generate(prompt());
  check(chat_result && chat_result->text == "Hello", "Chat Completions HTTP response");

  const auto local_result =
      cail::create_local({.endpoint = base + "/local/chat"})("test-model").generate(prompt());
  check(local_result && local_result->text == "Hello",
        "Local provider posts to the configured endpoint without an API key");
  auto shared_model = cail::create_local({.endpoint = base + "/local/chat"})("test-model");
  auto first = std::async(std::launch::async, [&] { return shared_model.generate(prompt()); });
  auto second = std::async(std::launch::async, [&] { return shared_model.generate(prompt()); });
  check(first.get() && second.get(), "one model handles concurrent default-transport calls");

  auto async_model = cail::create_local({.endpoint = base + "/chat/delay"})("test-model");
  check(async_model.adapter_capabilities().async_generation,
        "Chat Completions models report async generation support");
  std::vector<std::future<cail::Result<cail::GenerationResponse>>> pending;
  std::atomic<bool> observed_async_response{false};
  const auto async_began = std::chrono::steady_clock::now();
  for (int index = 0; index < 12; ++index) {
    auto completion = std::make_shared<std::promise<cail::Result<cail::GenerationResponse>>>();
    pending.push_back(completion->get_future());
    auto request = prompt();
    if (index == 0) {
      request.middleware.push_back({.after_response = [&](const cail::HttpResponse& response,
                                                          const cail::MiddlewareContext&) {
        observed_async_response.store(response.status_code == 200);
      }});
    }
    const auto started = async_model.generate_async(
        std::move(request), [completion](cail::Result<cail::GenerationResponse> response) {
          completion->set_value(std::move(response));
        });
    check(started.has_value(), "async generation starts");
  }
  const auto launch_time = std::chrono::steady_clock::now() - async_began;
  check(launch_time < std::chrono::milliseconds{800},
        "async generation returns before delayed HTTP responses");
  for (auto& result : pending) {
    const auto response = result.get();
    check(response && response->text == "Hello", "async generation decodes the response");
  }
  check(observed_async_response.load(), "async generation runs response middleware");

  auto error_model = cail::create_local({.endpoint = base + "/chat/error"})(
      "test-model", cail::make_default_http_transport({.retry = {.max_retries = 0}}));
  auto failed = std::make_shared<std::promise<cail::Result<cail::GenerationResponse>>>();
  auto failed_response = failed->get_future();
  const auto error_started = error_model.generate_async(
      prompt(), [failed](cail::Result<cail::GenerationResponse> response) {
        failed->set_value(std::move(response));
      });
  check(error_started.has_value(), "async error request starts");
  const auto async_error = failed_response.get();
  check(!async_error && async_error.error().code == cail::ErrorCode::http_status &&
            async_error.error().http_status == 429,
        "async generation preserves HTTP error context");

  auto held_model = cail::create_local({.endpoint = base + "/chat/hold"})("test-model");
  auto cancellation = std::make_shared<std::promise<cail::Result<cail::GenerationResponse>>>();
  auto cancelled_response = cancellation->get_future();
  std::stop_source async_stop;
  const auto active = held_model.generate_async(
      prompt(),
      [cancellation](cail::Result<cail::GenerationResponse> response) {
        cancellation->set_value(std::move(response));
      },
      async_stop.get_token());
  check(active.has_value(), "cancellable async generation starts");
  std::this_thread::sleep_for(std::chrono::milliseconds{50});
  async_stop.request_stop();
  check(cancelled_response.wait_for(std::chrono::seconds{2}) == std::future_status::ready,
        "async cancellation completes promptly");
  const auto cancelled_async = cancelled_response.get();
  check(!cancelled_async && cancelled_async.error().code == cail::ErrorCode::cancelled,
        "async generation reports cancellation");
  const std::vector<cail::LanguageModel> async_providers{
      cail::create_openai({.api_key = "test-key", .base_url = base + "/async"})("test-model"),
      cail::create_anthropic({.api_key = "test-key", .base_url = base + "/async/anthropic"})(
          "test-model"),
      cail::create_gemini({.api_key = "test-key", .base_url = base + "/async/gemini"})(
          "test-model"),
  };
  std::vector<std::future<cail::Result<cail::GenerationResponse>>> provider_results;
  const auto providers_began = std::chrono::steady_clock::now();
  for (const auto& model : async_providers) {
    auto completion = std::make_shared<std::promise<cail::Result<cail::GenerationResponse>>>();
    provider_results.push_back(completion->get_future());
    check(model
              .generate_async(
                  prompt(),
                  [completion](auto response) { completion->set_value(std::move(response)); })
              .has_value(),
          "native async provider starts over HTTP");
  }
  check(std::chrono::steady_clock::now() - providers_began < std::chrono::milliseconds{800},
        "native providers return before delayed responses");
  for (auto& response : provider_results) {
    check(response.wait_for(std::chrono::seconds{5}) == std::future_status::ready &&
              response.get().has_value(),
          "native async providers decode live HTTP responses");
  }
  for (const auto& model : async_providers) {
    std::promise<cail::Result<cail::GenerationResponse>> completion;
    auto result = completion.get_future();
    std::stop_source cancel;
    check(model
              .generate_async(
                  prompt(), [&](auto response) { completion.set_value(std::move(response)); },
                  cancel.get_token())
              .has_value(),
          "native cancellable HTTP request starts");
    cancel.request_stop();
    check(result.wait_for(std::chrono::seconds{2}) == std::future_status::ready,
          "native HTTP cancellation completes promptly");
    const auto response = result.get();
    check(!response && response.error().code == cail::ErrorCode::cancelled,
          "native async providers report HTTP cancellation");
  }
  const std::vector<cail::EmbeddingModel> async_embeddings{
      cail::create_openai({.api_key = "test-key", .base_url = base + "/async"})
          .embedding_model("test-model"),
      cail::create_gemini({.api_key = "test-key", .base_url = base + "/async/gemini"})
          .embedding_model("test-model"),
  };
  for (const auto& model : async_embeddings) {
    std::promise<cail::Result<cail::Embedding>> completion;
    auto result = completion.get_future();
    check(model.supports_async() &&
              model
                  .embed_async("one",
                               [&](auto response) { completion.set_value(std::move(response)); })
                  .has_value(),
          "native embedding provider starts async HTTP request");
    check(result.wait_for(std::chrono::milliseconds{50}) == std::future_status::timeout,
          "async embeddings return before delayed HTTP response");
    check(result.get().has_value(), "async embedding provider decodes live HTTP response");
    std::stop_source cancel;
    std::promise<cail::Result<cail::EmbeddingBatch>> cancelled;
    auto cancelled_result = cancelled.get_future();
    check(model
              .embed_many_async(
                  {"one", "two"}, [&](auto response) { cancelled.set_value(std::move(response)); },
                  cancel.get_token())
              .has_value(),
          "cancellable embedding HTTP request starts");
    cancel.request_stop();
    check(cancelled_result.wait_for(std::chrono::seconds{2}) == std::future_status::ready,
          "async embedding HTTP cancellation completes promptly");
    const auto response = cancelled_result.get();
    check(!response && response.error().code == cail::ErrorCode::cancelled,
          "embedding providers report HTTP cancellation");
  }
  auto count =
      cail::tool<CountInput, int>("count", "Count characters", [](const CountInput& input) {
        return static_cast<int>(input.query.size());
      });
  std::promise<cail::Result<cail::GenerationResponse>> loop_completion;
  auto loop_result = loop_completion.get_future();
  check(cail::generate_text_async(
            {.model = cail::create_local({.endpoint = base + "/async/tool-chat"})("test-model"),
             .prompt = "count abc",
             .tools = {count}},
            [&](auto response) { loop_completion.set_value(std::move(response)); })
            .has_value(),
        "async tool loop starts over live HTTP");
  const auto loop_response = loop_result.get();
  check(loop_response && loop_response->text == "Done" && loop_response->tool_results.size() == 1 &&
            loop_response->tool_results.front().output == "3",
        "async tool loop executes tools over live HTTP");

  // Direct transports own active I/O even after the caller destroys the wrapper.
  auto raw_completion = std::make_shared<std::promise<cail::Result<cail::HttpResponse>>>();
  auto raw_result = raw_completion->get_future();
  {
    cail::detail::GlazeHttpTransport transport;
    transport.send_async(
        {.url = base + "/chat/delay", .body = "{}"},
        [raw_completion](auto response) { raw_completion->set_value(std::move(response)); }, {});
  }
  check(raw_result.get().has_value(), "active HTTP owns its client after transport destruction");

  auto nested_completion = std::make_shared<std::promise<cail::Result<cail::GenerationResponse>>>();
  auto nested_result = nested_completion->get_future();
  auto reentrant = cail::create_local({.endpoint = base + "/chat"})("test-model");
  check(reentrant
            .generate_async(
                prompt(),
                [reentrant, nested_completion](auto response) {
                  check(response.has_value(), "outer async HTTP request succeeds");
                  const auto blocked = reentrant.generate(prompt());
                  check(!blocked && blocked.error().code == cail::ErrorCode::invalid_configuration,
                        "blocking generation inside an I/O callback fails instead of deadlocking");
                  check(reentrant
                            .generate_async(prompt(),
                                            [nested_completion](auto next) {
                                              nested_completion->set_value(std::move(next));
                                            })
                            .has_value(),
                        "an I/O callback can initiate another async request");
                })
            .has_value(),
        "reentrant callback test starts");
  check(nested_result.get().has_value(), "nested async HTTP request completes");

  std::string async_deltas;
  auto stream_completion = std::make_shared<std::promise<cail::Result<cail::GenerationResponse>>>();
  auto stream_result = stream_completion->get_future();
  check(
      cail::stream_text_async(
          {.model = reentrant, .prompt = "hello"},
          [&](const cail::StreamEvent& event) {
            if (const auto* text = std::get_if<cail::TextDelta>(&event)) {
              async_deltas += text->text;
              const auto blocked = reentrant.generate(prompt());
              check(
                  !blocked && blocked.error().code == cail::ErrorCode::invalid_configuration,
                  "blocking generation inside an HTTP event handler fails instead of deadlocking");
            }
          },
          [stream_completion](auto response) { stream_completion->set_value(std::move(response)); })
          .has_value(),
      "async streaming starts over live HTTP");
  const auto async_stream = stream_result.get();
  check(async_stream && async_stream->text == "Hi" && async_deltas == "Hi",
        "async streaming delivers live SSE events and its final response");

  auto cancelled_stream = std::make_shared<std::promise<cail::Result<cail::GenerationResponse>>>();
  auto cancelled_stream_result = cancelled_stream->get_future();
  std::stop_source stream_stop;
  check(
      cail::stream_text_async(
          {.model = held_model, .prompt = "hello", .tool_loop = {.stop = stream_stop.get_token()}},
          [&](const auto&) { stream_stop.request_stop(); },
          [cancelled_stream](auto response) { cancelled_stream->set_value(std::move(response)); })
          .has_value(),
      "cancellable async stream starts over live HTTP");
  const auto stopped_stream = cancelled_stream_result.get();
  check(!stopped_stream && stopped_stream.error().code == cail::ErrorCode::cancelled,
        "cancellation from an async SSE event ends the live stream");

  // One model multiplexes streams without a blocking-worker admission limit.
  std::vector<std::stop_source> stream_stops(8);
  std::vector<std::future<void>> connected;
  std::vector<std::future<cail::Result<cail::GenerationResponse>>> stream_results;
  for (auto& stop : stream_stops) {
    auto first_event = std::make_shared<std::promise<void>>();
    auto seen = std::make_shared<std::atomic<bool>>(false);
    connected.push_back(first_event->get_future());
    auto finished = std::make_shared<std::promise<cail::Result<cail::GenerationResponse>>>();
    stream_results.push_back(finished->get_future());
    check(held_model
              .stream_async(
                  prompt(),
                  [first_event, seen](const auto&) {
                    if (!seen->exchange(true))
                      first_event->set_value();
                  },
                  [finished](auto response) { finished->set_value(std::move(response)); },
                  stop.get_token())
              .has_value(),
          "concurrent native HTTP stream starts");
  }
  for (auto& ready : connected)
    check(ready.wait_for(std::chrono::seconds{5}) == std::future_status::ready,
          "all eight native streams connect before any stream finishes");
  for (auto& stop : stream_stops)
    stop.request_stop();
  for (auto& result : stream_results) {
    check(result.wait_for(std::chrono::seconds{2}) == std::future_status::ready,
          "native HTTP stream cancellation completes promptly");
    const auto response = result.get();
    check(!response && response.error().code == cail::ErrorCode::cancelled,
          "each multiplexed stream reports cancellation");
  }

  std::vector<cail::LanguageModel> native_stream_models{
      cail::create_openai({.api_key = "test-key", .base_url = base})("test-model"),
      cail::create_local({.endpoint = base + "/chat"})("test-model"),
      cail::create_anthropic({.api_key = "test-key", .base_url = base + "/anthropic"})(
          "test-model"),
      cail::create_gemini({.api_key = "test-key", .base_url = base + "/gemini"})("test-model"),
  };
  for (const auto& model : native_stream_models) {
    std::promise<cail::Result<cail::GenerationResponse>> finished;
    auto result = finished.get_future();
    std::string text;
    check(model.adapter_capabilities().async_streaming &&
              model
                  .stream_async(
                      prompt(),
                      [&](const auto& event) {
                        if (const auto* delta = std::get_if<cail::TextDelta>(&event))
                          text += delta->text;
                      },
                      [&](auto response) { finished.set_value(std::move(response)); })
                  .has_value(),
          "each provider family supports native asynchronous SSE");
    const auto response = result.get();
    check(response && response->text == "Hi" && text == "Hi",
          "native SSE decoding preserves provider events and final text");
  }

  auto thrown = std::make_shared<std::promise<void>>();
  auto thrown_result = thrown->get_future();
  check(reentrant
            .generate_async(prompt(),
                            [thrown](auto) {
                              thrown->set_value();
                              throw std::runtime_error("completion failed");
                            })
            .has_value(),
        "throwing HTTP completion starts");
  thrown_result.get();
  check(reentrant.generate(prompt()).has_value(),
        "I/O workers remain usable after a throwing completion");

  const auto keyed_local =
      cail::create_local({.api_key = "key", .endpoint = base + "/local/chat"})("test-model")
          .generate(prompt());
  check(!keyed_local && keyed_local.error().http_status == 401,
        "Local provider sends a configured API key");

  std::string chat_deltas;
  const auto chat_stream = chat.stream(prompt(), [&](const cail::StreamEvent& event) {
    if (const auto* delta = std::get_if<cail::TextDelta>(&event)) {
      chat_deltas += delta->text;
    }
  });
  check(chat_stream && chat_stream->text == "Hi" && chat_deltas == "Hi" && chat_stream->usage &&
            chat_stream->usage->input_tokens == 3,
        "Chat Completions SSE and usage");

  cail::detail::chat_completions::Client structured(
      {.endpoint = base + "/chat/structured", .api_key = "test-key"}, "test-model");
  auto structured_request = prompt();
  structured_request.structured_output =
      cail::StructuredOutput{.name = "answer", .schema = cail::schema<Answer>()};
  const auto structured_response = structured.generate(structured_request);
  check(structured_response && structured_response->text == R"({"answer":"yes"})",
        "Chat Completions sends structured output over HTTP");
  auto unsupported_request = prompt();
  unsupported_request.continuation_token = "response-id";
  const auto unsupported = chat.generate(unsupported_request);
  check(!unsupported && unsupported.error().code == cail::ErrorCode::invalid_configuration,
        "Chat Completions rejects provider continuation state");

  cail::detail::chat_completions::Client truncated_chat(
      {.endpoint = base + "/chat/truncated", .api_key = "test-key"}, "test-model");
  const auto truncated = truncated_chat.stream(prompt(), [](const cail::StreamEvent&) {});
  check(!truncated && truncated.error().code == cail::ErrorCode::provider_response,
        "Chat Completions rejects an interrupted stream");

  cail::detail::chat_completions::Client bad_chat(
      {.endpoint = base + "/chat/error", .api_key = "test-key"}, "test-model");
  const auto bad_result = bad_chat.generate(prompt());
  check(!bad_result && bad_result.error().code == cail::ErrorCode::http_status &&
            bad_result.error().http_status == 429 &&
            bad_result.error().provider_code == "rate_limited" &&
            bad_result.error().request_id == "req_local",
        "Chat Completions HTTP error context");
  const auto bad_stream = bad_chat.stream(prompt(), [](const cail::StreamEvent&) {});
  check(!bad_stream && bad_stream.error().http_status == 429 &&
            bad_stream.error().message == "slow down" &&
            bad_stream.error().provider_code == "rate_limited",
        "Chat Completions streaming HTTP error body");

  cail::detail::openai::Client responses(
      {.api_key = "test-key", .model = "test-model", .base_url = base});
  const auto response_result = responses.generate(prompt());
  check(response_result && response_result->text == "Hello", "OpenAI Responses HTTP response");
  std::string response_deltas;
  const auto response_stream = responses.stream(prompt(), [&](const cail::StreamEvent& event) {
    if (const auto* delta = std::get_if<cail::TextDelta>(&event)) {
      response_deltas += delta->text;
    }
  });
  check(response_stream && response_stream->text == "Hi" && response_deltas == "Hi",
        "OpenAI Responses SSE");

  cail::detail::openai::Client truncated_responses(
      {.api_key = "test-key", .model = "test-model", .base_url = base + "/truncated"});
  const auto truncated_response =
      truncated_responses.stream(prompt(), [](const cail::StreamEvent&) {});
  check(!truncated_response &&
            truncated_response.error().code == cail::ErrorCode::provider_response,
        "OpenAI Responses rejects an interrupted stream");

  cail::detail::anthropic::Client anthropic(
      {.api_key = "test-key", .model = "test-model", .base_url = base + "/anthropic"});
  const auto anthropic_result = anthropic.generate(prompt());
  check(anthropic_result && anthropic_result->text == "Searching" &&
            anthropic_result->reasoning == "Need facts" &&
            anthropic_result->tool_calls.size() == 1 &&
            anthropic_result->tool_calls[0].arguments.find("\"q\"") != std::string::npos &&
            anthropic_result->tool_calls[0].arguments.find("\"x\"") != std::string::npos &&
            anthropic_result->usage && anthropic_result->usage->cache_read_tokens == 2 &&
            anthropic_result->usage->cache_write_tokens == 1,
        "Anthropic HTTP maps text, reasoning, tools, and cache usage");
  std::vector<cail::StreamEvent> anthropic_events;
  const auto anthropic_stream = anthropic.stream(
      prompt(), [&](const cail::StreamEvent& event) { anthropic_events.push_back(event); });
  check(anthropic_stream && anthropic_stream->text == "Hi" &&
            anthropic_stream->tool_calls.size() == 1 &&
            anthropic_stream->tool_calls[0].arguments == R"({"q":"x"})" &&
            anthropic_stream->usage && anthropic_stream->usage->output_tokens == 4,
        "Anthropic SSE maps text, tool calls, and usage");
  check(anthropic_events.size() == 5 &&
            std::get_if<cail::ToolCallArgumentsDelta>(&anthropic_events[2]) &&
            std::get<cail::ToolCallArgumentsDelta>(anthropic_events[2]).output_index == 1 &&
            std::get_if<cail::ToolCallReady>(&anthropic_events[3]),
        "Anthropic SSE keeps tool indexes");

  cail::detail::anthropic::Client bad_anthropic(
      {.api_key = "test-key", .model = "test-model", .base_url = base + "/anthropic/error"});
  const auto anthropic_error = bad_anthropic.generate(prompt());
  check(!anthropic_error && anthropic_error.error().code == cail::ErrorCode::http_status &&
            anthropic_error.error().provider_type == "rate_limit_error" &&
            anthropic_error.error().request_id == "anthropic_req_local",
        "Anthropic HTTP error context");
  cail::detail::anthropic::Client truncated_anthropic(
      {.api_key = "test-key", .model = "test-model", .base_url = base + "/anthropic/truncated"});
  const auto interrupted_anthropic =
      truncated_anthropic.stream(prompt(), [](const cail::StreamEvent&) {});
  check(!interrupted_anthropic &&
            interrupted_anthropic.error().code == cail::ErrorCode::provider_response,
        "Anthropic rejects an interrupted stream");

  cail::detail::anthropic::Client validating_anthropic({.api_key = "test-key",
                                                        .model = "test-model",
                                                        .base_url = base + "/anthropic/validate",
                                                        .max_tokens = 256});
  auto mapping_request = cail::GenerationRequest{
      .messages =
          {
              cail::Message{.role = cail::MessageRole::system,
                            .content = {cail::TextPart{.text = "Be concise."}}},
              cail::Message{.role = cail::MessageRole::user,
                            .content = {cail::TextPart{.text = "Describe"},
                                        cail::ImagePart{.bytes = std::string{"A\0B", 3},
                                                        .mime_type = "image/png"}}},
              cail::Message{
                  .role = cail::MessageRole::assistant,
                  .tool_calls = {{.id = "toolu_1", .name = "search", .arguments = R"({"q":"x"})"}}},
              cail::Message{.role = cail::MessageRole::tool,
                            .content = {cail::TextPart{.text = "found"}},
                            .tool_call_id = "toolu_1"},
          },
      .tools = {cail::make_tool<Answer>("search", "Search")},
  };
  const auto mapped = validating_anthropic.generate(mapping_request);
  if (!mapped) {
    std::cerr << mapped.error().message << '\n';
  }
  check(mapped && mapped->text == "Done",
        "Anthropic request maps images, tool history, schema, and headers");
  cail::detail::anthropic::Client structured_anthropic(
      {.api_key = "test-key", .model = "test-model", .base_url = base + "/anthropic/structured"});
  auto anthropic_object_request = prompt();
  anthropic_object_request.structured_output =
      cail::StructuredOutput{.name = "answer", .schema = cail::schema<Answer>()};
  const auto anthropic_object = structured_anthropic.generate(anthropic_object_request);
  check(anthropic_object && anthropic_object->text == R"({"answer":"yes"})",
        "Anthropic sends a JSON Schema output format");
  mapping_request.messages[0].role = cail::MessageRole::developer;
  const auto unsupported_anthropic = validating_anthropic.generate(mapping_request);
  check(!unsupported_anthropic &&
            unsupported_anthropic.error().code == cail::ErrorCode::invalid_configuration,
        "Anthropic rejects unsupported developer-role messages");

  cail::detail::gemini::Client gemini(
      {.api_key = "test-key", .model = "test-model", .base_url = base + "/gemini"});
  const auto gemini_result = gemini.generate(prompt());
  check(gemini_result && gemini_result->text == "Hello", "Gemini HTTP response");
  std::string gemini_deltas;
  const auto gemini_stream = gemini.stream(prompt(), [&](const cail::StreamEvent& event) {
    if (const auto* delta = std::get_if<cail::TextDelta>(&event)) {
      gemini_deltas += delta->text;
    }
  });
  check(gemini_stream && gemini_stream->text == "Hi" && gemini_deltas == "Hi" &&
            gemini_stream->usage && gemini_stream->usage->input_tokens == 5,
        "Gemini SSE and usage");
  auto gemini_object_request = prompt();
  gemini_object_request.structured_output =
      cail::StructuredOutput{.name = "answer", .schema = cail::schema<Answer>()};
  const auto gemini_object = gemini.generate(gemini_object_request);
  check(gemini_object && gemini_object->text == R"({"answer":"yes"})", "Gemini JSON Schema output");
  auto gemini_tool_request = prompt();
  gemini_tool_request.tools = {cail::make_tool<Answer>("search", "Search")};
  const auto gemini_call = gemini.generate(gemini_tool_request);
  check(gemini_call && gemini_call->tool_calls.size() == 1 &&
            gemini_call->tool_calls.front().id == "call_1" &&
            gemini_call->tool_calls.front().provider_options.contains("thought_signature") &&
            gemini_call->tool_calls.front()
                    .provider_options.at("thought_signature")
                    .get<std::string>() == "signature" &&
            gemini_call->tool_calls.front().name == "search",
        "Gemini function call and thought signature");
  cail::detail::gemini::Client validating_gemini(
      {.api_key = "test-key", .model = "test-model", .base_url = base + "/gemini/validate"});
  cail::GenerationRequest gemini_history{
      .messages =
          {
              cail::Message{.role = cail::MessageRole::system,
                            .content = {cail::TextPart{.text = "Be concise."}}},
              cail::Message{.role = cail::MessageRole::user,
                            .content = {cail::ImagePart{.bytes = std::string{"A\0B", 3},
                                                        .mime_type = "image/png"}}},
              cail::Message{.role = cail::MessageRole::assistant,
                            .tool_calls = {cail::ToolCall{
                                .id = "call_1",
                                .name = "search",
                                .arguments = R"({"q":"x"})",
                                .provider_options = {{"thought_signature", "signature"}}}}},
              cail::Message{.role = cail::MessageRole::tool,
                            .content = {cail::TextPart{.text = R"({"answer":"yes"})"}},
                            .tool_call_id = "call_1"},
          },
  };
  const auto gemini_mapped = validating_gemini.generate(gemini_history);
  check(gemini_mapped && gemini_mapped->text == "Done",
        "Gemini maps images, tools, results, and thought signatures");

  cail::detail::anthropic::Client held_anthropic(
      {.api_key = "test-key", .model = "test-model", .base_url = base + "/anthropic/hold"});
  std::stop_source anthropic_stop;
  const auto anthropic_began = std::chrono::steady_clock::now();
  const auto cancelled_anthropic = held_anthropic.stream(
      prompt(),
      [&](const cail::StreamEvent& event) {
        if (std::holds_alternative<cail::TextDelta>(event)) {
          anthropic_stop.request_stop();
        }
      },
      anthropic_stop.get_token());
  check(!cancelled_anthropic && cancelled_anthropic.error().code == cail::ErrorCode::cancelled &&
            std::chrono::steady_clock::now() - anthropic_began < std::chrono::seconds(2),
        "Anthropic cancels the live HTTP stream");

  const auto openai_caps = cail::openai("test-model").adapter_capabilities();
  const auto chat_caps = cail::openrouter("test-model").adapter_capabilities();
  const auto anthropic_caps = cail::anthropic("test-model").adapter_capabilities();
  const auto gemini_caps = cail::gemini("test-model").adapter_capabilities();
  check(openai_caps.streaming && openai_caps.image_input && openai_caps.structured_output &&
            openai_caps.continuation && chat_caps.streaming && chat_caps.structured_output &&
            !chat_caps.continuation && anthropic_caps.streaming && anthropic_caps.tools &&
            anthropic_caps.structured_output && gemini_caps.streaming && gemini_caps.tools &&
            gemini_caps.structured_output && !gemini_caps.continuation,
        "Adapter capabilities reflect what each integration can encode and parse");

  cail::detail::chat_completions::Client held_chat(
      {.endpoint = base + "/chat/hold", .api_key = "test-key"}, "test-model");
  std::stop_source stop;
  const auto began = std::chrono::steady_clock::now();
  const auto cancelled = held_chat.stream(
      prompt(),
      [&](const cail::StreamEvent& event) {
        if (std::holds_alternative<cail::TextDelta>(event)) {
          stop.request_stop();
        }
      },
      stop.get_token());
  check(!cancelled && cancelled.error().code == cail::ErrorCode::cancelled &&
            std::chrono::steady_clock::now() - began < std::chrono::seconds(2),
        "Chat Completions cancels the live HTTP stream");

  coroutine_http(base);

  return failures == 0 ? 0 : 1;
}
