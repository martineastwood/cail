#include "test_support.hpp"

#include <cail/agent.hpp>
#include <cail/field.hpp>
#include <cail/json.hpp>
#include <cail/schema.hpp>
#include <cail/tool.hpp>

#include <algorithm>
#include <future>
#include <memory>

namespace test {

void test_control_character_json() {
  std::string controls;
  for (int ch = 0; ch < 32; ++ch) {
    controls += static_cast<char>(ch);
  }
  const auto encoded = cail::to_json(Address{.city = controls});
  check(encoded.has_value(), "control characters serialize");
  if (!encoded) {
    return;
  }
  const auto decoded = cail::from_json<Address>(*encoded);
  check(decoded && decoded->city == controls, "JSON preserves every control character");
  const auto merged = cail::merge_json_objects(*encoded, {{"extra", std::string(1, '\0')}});
  check(merged.has_value(), "provider options merge with control characters");
  if (!merged) {
    return;
  }
  glz::generic parsed;
  const auto error = glz::read_json(parsed, *merged);
  check(!error && parsed["city"].get<std::string>() == controls &&
            parsed["extra"].get<std::string>() == std::string(1, '\0'),
        "merged JSON preserves control characters in both objects");
}

void test_field_value_api() {
  cail::Field<double> confidence{
      .value = 0.5,
      .description = "Confidence score",
      .minimum = 0.0,
      .maximum = 1.0,
  };
  check(confidence.value == 0.5, "field runtime value is read through .value");
  confidence = 0.8;
  check(confidence.value == 0.8, "field assignment updates its runtime value");
  check(confidence.description == "Confidence score" && confidence.minimum == 0.0 &&
            confidence.maximum == 1.0,
        "field assignment preserves schema metadata");
}

void test_optional_schema_and_json() {
  const auto output_schema = cail::schema<Profile>();
  check(output_schema.properties.has_value(), "schema exposes model properties");
  check(output_schema.required.has_value(), "schema exposes required properties");
  if (!output_schema.properties || !output_schema.required) {
    return;
  }

  check(std::ranges::find(*output_schema.required, "answer") != output_schema.required->end(),
        "non-optional property remains required");
  check(std::ranges::find(*output_schema.required, "confidence") == output_schema.required->end(),
        "optional property stays optional in the generic schema");
  const auto& confidence = *output_schema.properties->at("confidence");
  const auto* confidence_types = std::get_if<std::vector<cail::SchemaType>>(&confidence.type);
  check(confidence_types != nullptr, "optional property schema uses a type union");
  if (confidence_types != nullptr) {
    check(*confidence_types ==
              std::vector<cail::SchemaType>{cail::SchemaType::integer, cail::SchemaType::null},
          "optional integer schema allows null");
  }

  auto missing = cail::from_json<Profile>(R"({"answer":"ok"})");
  check(missing.has_value(), "missing optional properties deserialize");
  if (missing) {
    check(!missing->confidence && !missing->address,
          "missing optional values become empty optionals");
  }

  auto null_value = cail::from_json<Profile>(R"({"answer":"ok","confidence":null,"address":null})");
  check(null_value.has_value(), "explicit null values deserialize");
  if (null_value) {
    check(!null_value->confidence && !null_value->address,
          "explicit null values become empty optionals");
  }

  Profile populated{
      .answer = "done",
      .confidence = 7,
      .address = Address{.city = "London"},
  };
  auto json = cail::to_json(populated);
  check(json && json->find(R"("confidence":7)") != std::string::npos,
        "populated optional values serialize normally");
}

void test_tool_loop() {
  std::vector<std::size_t> observed_steps;
  ScriptedClient client({
      cail::GenerationResponse{
          .tool_calls = {cail::ToolCall{
              .id = "call-1", .name = "count", .arguments = R"({"query":"abc"})"}},
          .continuation_token = "turn-1",
      },
      cail::GenerationResponse{.text = "counted"},
  });
  auto count_tool = cail::tool<ToolInput, ToolOutput>(
      "count", "Count characters", [](const ToolInput& input, const cail::ToolContext& context) {
        check(context.call_id == "call-1" && context.round == 0, "tool receives call context");
        return ToolOutput{.count = static_cast<int>(input.query.size())};
      });

  const auto result = cail::run_tool_loop(
      client,
      cail::GenerationRequest{
          .messages = {cail::Message{.content = {cail::TextPart{.text = "count abc"}}}},
          .session_id = "stable-session",
          .middleware = {cail::GenerationMiddleware{
              .after_step =
                  [&](const cail::Result<cail::GenerationResponse>& response,
                      const cail::MiddlewareContext& context) {
                    check(response.has_value(), "step middleware observes successful results");
                    observed_steps.push_back(context.step);
                  },
          }},
      },
      std::vector<cail::Tool>{count_tool});
  check(result && result->text == "counted", "tool loop returns the final model response");
  check(result && result->tool_results.size() == 1, "tool loop records tool output");
  check(result && result->tool_results.front().output == R"({"count":3})",
        "tool output stays typed and serializes");
  check(client.requests.size() == 2, "tool loop makes a follow-up model call");
  check(client.requests.size() == 2 && client.requests[1].continuation_token == "turn-1",
        "tool loop carries continuation state");
  check(client.requests.size() == 2 && client.requests[1].session_id == "stable-session",
        "tool loop carries the caller session ID");
  check(client.requests.size() == 2 && client.requests[1].messages.front().tool_call_id == "call-1",
        "tool loop sends the result for the matching call");
  check(observed_steps == std::vector<std::size_t>{0, 1},
        "middleware observes every model step in a tool loop");
}

void test_agent() {
  auto client = std::make_shared<ScriptedClient>(
      std::vector<cail::GenerationResponse>{cail::GenerationResponse{.text = "hello"}});
  cail::Agent agent({
      .model = cail::LanguageModel(
          [client](const cail::GenerationRequest& request, std::stop_token stop) {
            return client->generate(request, stop);
          }),
      .instructions = "Be concise.",
      .tools = {cail::tool<ToolInput, ToolOutput>("count", "Count characters",
                                                  [](const ToolInput& input) {
                                                    return ToolOutput{.count = static_cast<int>(
                                                                          input.query.size())};
                                                  })},
  });

  const auto response = agent.generate("Say hello.");
  check(response && response->text == "hello", "agent returns the model response");
  check(client->requests.size() == 1 && client->requests.front().messages.size() == 2,
        "agent combines its instructions with the prompt");
  check(client->requests.front().messages.front().role == cail::MessageRole::system &&
            std::get<cail::TextPart>(client->requests.front().messages.front().content.front())
                    .text == "Be concise.",
        "agent sends its instructions as the first system message");
  check(client->requests.front().tools.size() == 1 &&
            client->requests.front().tools.front().name == "count",
        "agent offers its tools on every run");
}

void test_streaming_tool_loop_and_stop_conditions() {
  ScriptedClient client({
      cail::GenerationResponse{
          .text = "checking",
          .tool_calls = {cail::ToolCall{
              .id = "call-1", .name = "count", .arguments = R"({"query":"abc"})"}},
          .provider_options = {{"state", "kept"}},
      },
      cail::GenerationResponse{.text = "counted"},
  });
  auto count_tool =
      cail::tool<ToolInput, ToolOutput>("count", "Count characters", [](const ToolInput& input) {
        return ToolOutput{.count = static_cast<int>(input.query.size())};
      });
  std::string streamed;
  const auto response =
      cail::stream_tool_loop(client, cail::GenerationRequest{}, std::vector<cail::Tool>{count_tool},
                             [&](const cail::StreamEvent& event) {
                               if (const auto* delta = std::get_if<cail::TextDelta>(&event)) {
                                 streamed += delta->text;
                               }
                             });
  check(response && response->text == "counted" && streamed == "checkingcounted",
        "streaming tool loop forwards events from every model step");
  check(response && response->tool_results.size() == 1,
        "streaming tool loop executes tools and returns their results");
  check(client.requests.size() == 2 && client.requests[1].messages.size() == 2 &&
            client.requests[1].messages.front().provider_options.contains("state"),
        "tool loop preserves provider metadata on assistant history");

  ScriptedClient stopped_client({cail::GenerationResponse{
      .tool_calls = {cail::ToolCall{
          .id = "call-1", .name = "count", .arguments = R"({"query":"abc"})"}},
  }});
  const auto stopped = cail::run_tool_loop(
      stopped_client, {}, std::vector<cail::Tool>{count_tool},
      {.stop_when = [](const cail::GenerationResponse&, const cail::MiddlewareContext& context) {
        return context.step == 0;
      }});
  check(stopped && stopped->tool_calls.size() == 1 && stopped_client.requests.size() == 1,
        "custom stop condition returns before another tool-loop step");

  std::stop_source cancelled;
  cancelled.request_stop();
  const auto cancelled_response =
      cail::stream_tool_loop(stopped_client, {}, std::vector<cail::Tool>{count_tool},
                             [](const cail::StreamEvent&) {}, {.stop = cancelled.get_token()});
  check(!cancelled_response && cancelled_response.error().code == cail::ErrorCode::cancelled,
        "streaming tool loop honors cancellation before the first step");
}

void test_transport_retries() {
  class FlakyTransport final : public cail::HttpTransport {
  public:
    [[nodiscard]] cail::Result<cail::HttpResponse> send(const cail::HttpRequest&,
                                                        std::stop_token) override {
      ++attempts;
      return cail::HttpResponse{.status_code = attempts < 3 ? 429 : 200};
    }

    [[nodiscard]] cail::Result<cail::HttpResponse> stream(const cail::HttpRequest&,
                                                          const cail::HttpDataHandler& on_data,
                                                          std::stop_token) override {
      ++attempts;
      if (attempts == 1) {
        return cail::HttpResponse{.status_code = 503};
      }
      on_data("done");
      return cail::HttpResponse{.status_code = 200};
    }

    int attempts{};
  };

  auto send_transport = std::make_unique<FlakyTransport>();
  auto* send_flaky = send_transport.get();
  cail::RetryingHttpTransport retrying_send(std::move(send_transport),
                                            {.max_retries = 2, .initial_delay = {}});
  const auto response = retrying_send.send({}, {});
  check(response && response->status_code == 200 && send_flaky->attempts == 3,
        "transport retries 429 responses up to the configured limit");

  auto stream_transport = std::make_unique<FlakyTransport>();
  auto* stream_flaky = stream_transport.get();
  cail::RetryingHttpTransport retrying_stream(std::move(stream_transport),
                                              {.max_retries = 1, .initial_delay = {}});
  std::string data;
  const auto stream_response =
      retrying_stream.stream({}, [&](std::string_view chunk) { data.append(chunk); }, {});
  check(stream_response && data == "done" && stream_flaky->attempts == 2,
        "transport retries a 5xx stream before delivering data");

  class RetryWaitTransport final : public cail::HttpTransport {
  public:
    [[nodiscard]] cail::Result<cail::HttpResponse> send(const cail::HttpRequest&,
                                                        std::stop_token) override {
      sent.set_value();
      return cail::HttpResponse{.status_code = 503};
    }
    [[nodiscard]] cail::Result<cail::HttpResponse>
    stream(const cail::HttpRequest&, const cail::HttpDataHandler&, std::stop_token) override {
      return std::unexpected(cail::Error{.code = cail::ErrorCode::invalid_configuration});
    }
    std::promise<void> sent;
  };
  auto retry_wait_transport = std::make_unique<RetryWaitTransport>();
  auto sent = retry_wait_transport->sent.get_future();
  cail::RetryingHttpTransport retry_wait(
      std::move(retry_wait_transport),
      {.max_retries = 2, .initial_delay = std::chrono::seconds{5}});
  std::stop_source stop;
  auto pending =
      std::async(std::launch::async, [&] { return retry_wait.send({}, stop.get_token()); });
  sent.wait();
  const auto began = std::chrono::steady_clock::now();
  stop.request_stop();
  const auto cancelled = pending.get();
  check(!cancelled && cancelled.error().code == cail::ErrorCode::cancelled &&
            std::chrono::steady_clock::now() - began < std::chrono::seconds{2},
        "cancellation interrupts the HTTP retry delay");

  class TimeoutTransport final : public cail::HttpTransport {
  public:
    [[nodiscard]] cail::Result<cail::HttpResponse> send(const cail::HttpRequest& request,
                                                        std::stop_token) override {
      timeout = request.timeout;
      return cail::HttpResponse{.status_code = 200};
    }
    [[nodiscard]] cail::Result<cail::HttpResponse>
    stream(const cail::HttpRequest&, const cail::HttpDataHandler&, std::stop_token) override {
      return cail::HttpResponse{.status_code = 200};
    }
    std::chrono::seconds timeout{};
  };
  auto timeout_transport = std::make_unique<TimeoutTransport>();
  auto* timeout_probe = timeout_transport.get();
  cail::TimeoutHttpTransport configured_timeout(std::move(timeout_transport),
                                                std::chrono::seconds{7});
  static_cast<void>(configured_timeout.send({}, {}));
  check(timeout_probe->timeout == std::chrono::seconds{7},
        "transport applies the configured request timeout");
}

void test_async_transport_retries() {
  class AsyncTransport final : public cail::HttpTransport {
  public:
    [[nodiscard]] cail::Result<cail::HttpResponse> send(const cail::HttpRequest&,
                                                        std::stop_token) override {
      return std::unexpected(cail::Error{.code = cail::ErrorCode::invalid_configuration});
    }
    void send_async(cail::HttpRequest, cail::HttpCompletion complete, std::stop_token) override {
      ++attempts;
      complete(cail::HttpResponse{.status_code = attempts < 3 ? 503 : 200});
    }
    [[nodiscard]] cail::Result<cail::HttpResponse>
    stream(const cail::HttpRequest&, const cail::HttpDataHandler&, std::stop_token) override {
      return std::unexpected(cail::Error{.code = cail::ErrorCode::invalid_configuration});
    }
    int attempts{};
  };

  auto transport = std::make_unique<AsyncTransport>();
  auto* probe = transport.get();
  cail::RetryingHttpTransport retrying(std::move(transport),
                                       {.max_retries = 2, .initial_delay = {}});
  std::promise<cail::Result<cail::HttpResponse>> completed;
  auto result = completed.get_future();
  retrying.send_async(
      {},
      [&](cail::Result<cail::HttpResponse> response) { completed.set_value(std::move(response)); },
      {});
  const auto response = result.get();
  check(response && response->status_code == 200 && probe->attempts == 3,
        "async transport retries 5xx responses without blocking the caller");

  auto cancel_transport = std::make_unique<AsyncTransport>();
  cail::RetryingHttpTransport cancel_retry(
      std::move(cancel_transport), {.max_retries = 2, .initial_delay = std::chrono::seconds{5}});
  std::promise<cail::Result<cail::HttpResponse>> cancelled;
  auto cancelled_result = cancelled.get_future();
  std::stop_source stop;
  cancel_retry.send_async(
      {}, [&](cail::Result<cail::HttpResponse> value) { cancelled.set_value(std::move(value)); },
      stop.get_token());
  stop.request_stop();
  check(cancelled_result.wait_for(std::chrono::seconds{1}) == std::future_status::ready,
        "cancellation interrupts an async retry delay");
  const auto stopped = cancelled_result.get();
  check(!stopped && stopped.error().code == cail::ErrorCode::cancelled,
        "async retry reports cancellation");
}

} // namespace test

int main() {
  test::test_control_character_json();
  test::test_field_value_api();
  test::test_optional_schema_and_json();
  test::test_tool_loop();
  test::test_agent();
  test::test_streaming_tool_loop_and_stop_conditions();
  test::test_transport_retries();
  test::test_async_transport_retries();
  return test::failures == 0 ? 0 : 1;
}
