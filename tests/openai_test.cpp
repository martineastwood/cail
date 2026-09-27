#include "test_support.hpp"

#include <cail/openai.hpp>

#include <stop_token>

namespace test {

void test_openai_strict_optional_schemas() {
  auto transport = std::make_unique<StubTransport>();
  auto *stub = transport.get();
  cail::detail::openai::Client client(
      {.api_key = "test-key", .model = "test-model"}, std::move(transport));

  const cail::GenerationRequest request{
      .messages = {cail::Message{
          .role = cail::MessageRole::user,
          .content = {cail::TextPart{.text = "make a profile"}},
      }},
      .tools = {cail::make_tool<ToolInput>("search", "Search for results")},
      .structured_output =
          cail::StructuredOutput{.schema = cail::schema<Profile>()},
  };
  const auto response = client.generate(request);
  check(response.has_value(),
        "OpenAI accepts optional output and tool schemas");
  check(stub->request.body.find(
            R"("required":["address","answer","confidence"])") !=
            std::string::npos,
        "OpenAI makes optional output properties required");
  check(stub->request.body.find(
            R"("confidence":{"type":["integer","null"]})") != std::string::npos,
        "OpenAI encodes optional output properties as nullable");
  check(stub->request.body.find(R"("required":["limit","query"])") !=
            std::string::npos,
        "OpenAI makes optional tool arguments required");
  check(stub->request.body.find(R"("limit":{"type":["integer","null"]})") !=
            std::string::npos,
        "OpenAI encodes optional tool arguments as nullable");
  check(stub->request.body.find(
            R"("address":{"type":["object","null"],"properties":{"city")") !=
            std::string::npos,
        "OpenAI preserves nested schemas on nullable object properties");
  check(stub->request.body.find(R"("required":["city"])") != std::string::npos,
        "OpenAI still requires nested object properties");
}

void test_openai_reasoning_summary() {
  auto transport = std::make_unique<StubTransport>();
  transport->response.body =
      R"({"status":"completed","output":[{"type":"reasoning","summary":[{"type":"summary_text","text":"checked the result"}]},{"type":"message","content":[{"type":"output_text","text":"Ready"}]}],"usage":{"input_tokens":20,"output_tokens":8,"input_tokens_details":{"cached_tokens":12},"output_tokens_details":{"reasoning_tokens":3}}})";
  cail::detail::openai::Client client(
      {.api_key = "test-key", .model = "test-model"}, std::move(transport));
  const auto response = client.generate("hello");
  check(response && response->reasoning == "checked the result" &&
            response->text == "Ready",
        "OpenAI returns reasoning summaries alongside text");
  check(response && response->usage &&
            response->usage->cache_read_tokens == 12 &&
            response->usage->reasoning_tokens == 3,
        "OpenAI preserves reported cache and reasoning usage");
}

void test_streaming_across_chunk_boundaries() {
  auto transport = std::make_unique<StubTransport>();
  auto *stub = transport.get();
  stub->response.body.clear();
  stub->chunks = {
      "event: response.output_text.delta\ndata: "
      "{\"type\":\"response.output_text.delta\",\"delta\":\"hel",
      "lo\"}\n\n",
      "event: response.completed\ndata: "
      "{\"type\":\"response.completed\",\"response\":{\"status\":\"completed\","
      "\"output\":[],\"usage\":{\"input_tokens\":2,\"output_tokens\":1}}}\n\n",
  };
  cail::detail::openai::Client client(
      {.api_key = "test-key", .model = "test-model"}, std::move(transport));

  std::string text;
  const auto response =
      client.stream("say hello", [&](const cail::StreamEvent &event) {
        if (const auto *delta = std::get_if<cail::TextDelta>(&event)) {
          text += delta->text;
        }
      });
  check(response.has_value(), "split SSE stream completes successfully");
  check(text == "hello", "split SSE text deltas are reassembled");
  check(response && response->usage && response->usage->input_tokens == 2,
        "stream response includes final usage");
  check(stub->request.url == "https://api.openai.com/v1/responses",
        "stream request uses the responses endpoint");
}

void test_normalized_stream_events() {
  auto transport = std::make_unique<StubTransport>();
  auto *stub = transport.get();
  stub->response.body.clear();
  stub->chunks = {
      "event: response.reasoning_summary_text.delta\ndata: "
      "{\"delta\":\"thinking\"}\n\n",
      "event: response.output_text.delta\ndata: {\"delta\":\"Hello\"}\n\n",
      "event: response.function_call_arguments.delta\ndata: "
      "{\"output_index\":1,\"delta\":\"{\\\"q\\\":\"}\n\n",
      "event: response.completed\ndata: "
      "{\"response\":{\"status\":\"completed\",\"output\":[{\"type\":"
      "\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"Hello\"}]}"
      ",{\"type\":\"function_call\",\"call_id\":\"call-1\",\"name\":\"search\","
      "\"arguments\":\"{\\\"q\\\":\\\"x\\\"}\"}],\"usage\":{\"input_tokens\":2,"
      "\"output_tokens\":3}}}\n\n",
  };
  cail::detail::openai::Client client(
      {.api_key = "test-key", .model = "test-model"}, std::move(transport));

  std::vector<cail::StreamEvent> events;
  const auto response = client.stream(
      "Say hello and search",
      [&](const cail::StreamEvent &event) { events.push_back(event); });
  check(response && response->text == "Hello" &&
            response->reasoning == "thinking",
        "streaming returns final text and reasoning");
  check(response && response->tool_calls.size() == 1 &&
            response->tool_calls.front().id == "call-1",
        "streaming returns the completed tool call");
  check(events.size() == 5, "streaming emits each normalized model event");
  if (events.size() == 5) {
    const auto *reasoning = std::get_if<cail::ReasoningDelta>(&events[0]);
    const auto *text = std::get_if<cail::TextDelta>(&events[1]);
    const auto *arguments =
        std::get_if<cail::ToolCallArgumentsDelta>(&events[2]);
    const auto *call = std::get_if<cail::ToolCallReady>(&events[3]);
    const auto *usage = std::get_if<cail::UsageUpdate>(&events[4]);
    check(reasoning && reasoning->text == "thinking",
          "reasoning delta is normalized");
    check(text && text->text == "Hello", "text delta is normalized");
    check(arguments && arguments->output_index == 1 &&
              arguments->arguments == R"({"q":)",
          "tool argument delta keeps its output index");
    check(call && call->output_index == 1 && call->call.name == "search",
          "completed tool call keeps its output index");
    check(usage && usage->usage.input_tokens == 2 &&
              usage->usage.output_tokens == 3,
          "final usage is normalized");
  }
}

void test_stream_cancellation() {
  auto transport = std::make_unique<StubTransport>();
  transport->chunks = {
      "event: response.output_text.delta\ndata: {\"delta\":\"first\"}\n\n",
      "event: response.output_text.delta\ndata: {\"delta\":\"second\"}\n\n",
  };
  cail::detail::openai::Client client(
      {.api_key = "test-key", .model = "test-model"}, std::move(transport));
  std::stop_source cancellation;
  std::string text;
  const auto response = client.stream(
      "Say something",
      [&](const cail::StreamEvent &event) {
        if (const auto *delta = std::get_if<cail::TextDelta>(&event)) {
          text += delta->text;
          cancellation.request_stop();
        }
      },
      cancellation.get_token());
  check(!response && response.error().code == cail::ErrorCode::cancelled,
        "cancelling during a callback returns a cancellation error");
  check(text == "first", "cancellation prevents later events");
}

void test_http_error_mapping() {
  auto transport = std::make_unique<StubTransport>();
  transport->response = cail::HttpResponse{
      .status_code = 401,
      .body =
          R"({"error":{"message":"bad key","code":"invalid_api_key","type":"authentication_error"}})",
      .headers = {{.name = "X-Request-Id", .value = "req_123"}},
  };
  cail::detail::openai::Client client(
      {.api_key = "test-key", .model = "test-model"}, std::move(transport));
  const auto response = client.generate("hello");
  check(!response, "non-success HTTP status returns an error");
  check(!response && response.error().code == cail::ErrorCode::http_status,
        "non-success HTTP status maps to the HTTP error code");
  check(!response && response.error().http_status == 401,
        "HTTP status is retained on the error");
  check(!response &&
            response.error().message.find("bad key") != std::string::npos,
        "provider error message is retained");
  check(!response && response.error().provider_code == "invalid_api_key" &&
            response.error().provider_type == "authentication_error" &&
            response.error().request_id == "req_123",
        "provider error context is retained");
}

void test_image_content() {
  auto transport = std::make_unique<StubTransport>();
  auto *stub = transport.get();
  cail::detail::openai::Client client(
      {.api_key = "test-key", .model = "test-model"}, std::move(transport));

  const auto response = client.generate(cail::GenerationRequest{
      .messages = {cail::Message{
          .role = cail::MessageRole::user,
          .content =
              {
                  cail::TextPart{.text = "Inspect this image."},
                  cail::ImagePart{.bytes = std::string{"A\0B", 3},
                                  .mime_type = "image/png"},
              },
      }},
  });
  check(response.has_value(), "OpenAI accepts user text and image content");
  const auto text_position = stub->request.body.find("Inspect this image.");
  const auto image_position =
      stub->request.body.find("data:image/png;base64,QQBC");
  check(text_position != std::string::npos &&
            image_position != std::string::npos &&
            text_position < image_position,
        "OpenAI preserves content order and encodes original image bytes");

  const auto invalid = client.generate(cail::GenerationRequest{
      .messages = {cail::Message{
          .role = cail::MessageRole::assistant,
          .content = {cail::ImagePart{.bytes = "image",
                                      .mime_type = "image/png"}},
      }},
  });
  check(!invalid &&
            invalid.error().code == cail::ErrorCode::invalid_configuration,
        "OpenAI rejects image content in unsupported roles");
}

} // namespace test

int main() {
  test::test_openai_strict_optional_schemas();
  test::test_openai_reasoning_summary();
  test::test_streaming_across_chunk_boundaries();
  test::test_normalized_stream_events();
  test::test_stream_cancellation();
  test::test_http_error_mapping();
  test::test_image_content();
  return test::failures == 0 ? 0 : 1;
}
