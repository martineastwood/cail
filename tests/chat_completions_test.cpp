#include "test_support.hpp"

#include <cail/chat_completions.hpp>

namespace test {

void test_chat_completions() {
  auto transport = std::make_unique<StubTransport>();
  auto* stub = transport.get();
  stub->response.body =
      R"({"choices":[{"index":0,"finish_reason":"tool_calls","message":{"content":"Searching","tool_calls":[{"id":"call-1","type":"function","function":{"name":"search","arguments":"{\"q\":\"x\"}"}}]}}],"usage":{"prompt_tokens":20,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":10}}})";
  cail::detail::chat_completions::Client client(
      "https://example.test/v1/chat/completions", "test-model", "test-key",
      {{.name = "X-App", .value = "cail"}}, std::move(transport));
  const auto response = client.generate(cail::GenerationRequest{
      .messages = {cail::Message{.role = cail::MessageRole::user,
                                 .content = {cail::TextPart{.text = "Find x"}}}},
      .tools = {cail::make_tool<ToolInput>("search", "Search")},
  });
  check(response && response->text == "Searching" && response->tool_calls.size() == 1 &&
            response->tool_calls[0].name == "search",
        "Chat Completions decodes tool calls");
  check(response && response->usage && response->usage->cache_read_tokens == 10,
        "Chat Completions decodes usage");
  check(stub->request.url == "https://example.test/v1/chat/completions" &&
            stub->request.body.find("\"model\":\"test-model\"") != std::string::npos &&
            stub->request.body.find("\"tools\"") != std::string::npos,
        "Chat Completions sends the configured endpoint, model, and tools");
}

void test_chat_completions_stream() {
  auto transport = std::make_unique<StubTransport>();
  auto* stub = transport.get();
  stub->chunks = {
      "data: "
      "{\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\",\"tool_calls\":"
      "[{\"index\":1,\"id\":\"call-1\",\"function\":{\"name\":\"search\","
      "\"arguments\":\"{\\\"q\\\":\"}}]}}]}\n\n",
      "data: "
      "{\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":1,"
      "\"function\":{\"arguments\":\"\\\"x\\\"}\"}}]},\"finish_reason\":\"tool_"
      "calls\"}]}\n\n",
      "data: "
      "{\"choices\":[],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":2}}"
      "\n\n",
      "data: [DONE]\n\n",
  };
  cail::detail::chat_completions::Client client("https://example.test/v1/chat/completions",
                                                "test-model", "test-key", {}, std::move(transport));
  std::vector<cail::StreamEvent> events;
  const auto response = client.stream(
      cail::GenerationRequest{
          .messages = {cail::Message{.role = cail::MessageRole::user,
                                     .content = {cail::TextPart{.text = "Hi"}}}},
      },
      [&](const cail::StreamEvent& event) { events.push_back(event); });
  check(response && response->text == "Hi" && response->tool_calls.size() == 1 &&
            response->tool_calls[0].arguments == R"({"q":"x"})",
        "Chat Completions assembles streamed calls");
  check(response && response->usage && response->usage->input_tokens == 4,
        "Chat Completions receives final streamed usage");
  check(stub->request.body.find("\"include_usage\":true") != std::string::npos,
        "Chat Completions requests streamed usage");
  check(events.size() == 5 && std::get_if<cail::ToolCallArgumentsDelta>(&events[1]) &&
            std::get<cail::ToolCallArgumentsDelta>(events[1]).output_index == 1 &&
            std::get_if<cail::ToolCallReady>(&events[4]),
        "Chat Completions preserves tool indexes in events");
}

void test_chat_completions_mistral_content() {
  auto transport = std::make_unique<StubTransport>();
  transport->chunks = {
      "data: "
      "{\"choices\":[{\"index\":0,\"delta\":{\"content\":[{\"type\":\"thinking\",\"thinking\":[{"
      "\"type\":\"text\",\"text\":\"First step\"}]}]}}]}\n\n",
      "data: "
      "{\"choices\":[{\"index\":0,\"delta\":{\"content\":[{\"type\":\"thinking\",\"thinking\":[{"
      "\"type\":\"text\",\"text\":\" next\"}]},{\"type\":\"text\",\"text\":\"Answer\"}]}}]}\n\n",
      "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\" "
      "done\"},\"finish_reason\":\"stop\"}]}\n\n",
      "data: [DONE]\n\n",
  };
  cail::detail::chat_completions::Client client("https://example.test/v1/chat/completions",
                                                "test-model", "test-key", {}, std::move(transport));
  std::vector<cail::StreamEvent> events;
  const auto response = client.stream(
      cail::GenerationRequest{
          .messages = {cail::Message{.role = cail::MessageRole::user,
                                     .content = {cail::TextPart{.text = "Hi"}}}}},
      [&](const cail::StreamEvent& event) { events.push_back(event); });
  check(response && response->reasoning == "First step next" && response->text == "Answer done",
        "Chat Completions decodes Mistral thinking and text parts");
  check(events.size() >= 3 && std::get_if<cail::ReasoningDelta>(&events[0]) &&
            std::get_if<cail::TextDelta>(&events[2]),
        "Chat Completions emits Mistral reasoning before answer text");
}

void test_chat_completions_history_and_image() {
  auto transport = std::make_unique<StubTransport>();
  auto* stub = transport.get();
  stub->response.body =
      R"({"choices":[{"index":0,"finish_reason":"stop","message":{"content":"Done"}}]})";
  cail::detail::chat_completions::Client client("https://example.test/v1/chat/completions",
                                                "test-model", "test-key", {}, std::move(transport));
  const auto
      response = client
                     .generate(
                         cail::GenerationRequest{
                             .messages =
                                 {
                                     cail::Message{
                                         .role = cail::MessageRole::user,
                                         .content = {cail::TextPart{.text = "Describe"},
                                                     cail::ImagePart{.bytes = std::string{"A\0B", 3}, .mime_type = "image/png"}}},
                                     cail::Message{.role = cail::MessageRole::assistant,
                                                   .tool_calls = {{.id = "call-1",
                                                                   .name = "search",
                                                                   .arguments = "{}"}}},
                                     cail::
                                         Message{.role = cail::MessageRole::tool,
                                                 .content = {cail::TextPart{.text = "found"}},
                                                 .tool_call_id = "call-1"},
                                 },
                         });
  check(response && response->text == "Done", "Chat Completions accepts image and tool history");
  check(stub->request.body.find("data:image/png;base64,QQBC") != std::string::npos &&
            stub->request.body.find("\"tool_call_id\":\"call-1\"") != std::string::npos &&
            stub->request.body.find("\"tool_calls\"") != std::string::npos &&
            stub->request.body.find("\"index\"") == std::string::npos,
        "Chat Completions encodes images and matching tool history");
}

void test_chat_completions_structured_output() {
  auto transport = std::make_unique<StubTransport>();
  auto* stub = transport.get();
  stub->response.body =
      R"({"choices":[{"index":0,"finish_reason":"stop","message":{"content":"{\"answer\":\"yes\",\"confidence\":null,\"address\":null}"}}]})";
  cail::detail::chat_completions::Client client("https://example.test/v1/chat/completions",
                                                "test-model", "test-key", {}, std::move(transport));
  const auto response = client.generate(cail::GenerationRequest{
      .messages = {cail::Message{.content = {cail::TextPart{.text = "Classify this."}}}},
      .structured_output =
          cail::StructuredOutput{.name = "profile", .schema = cail::schema<Profile>()},
  });
  check(response && response->text.find(R"("answer":"yes")") != std::string::npos,
        "Chat Completions returns structured JSON text");
  check(stub->request.body.find("\"response_format\":{\"type\":\"json_schema\"") !=
                std::string::npos &&
            stub->request.body.find("\"name\":\"profile\"") != std::string::npos &&
            stub->request.body.find("\"strict\":true") != std::string::npos,
        "Chat Completions sends a strict JSON Schema response format");
  check(stub->request.body.find("\"confidence\":{\"type\":[\"integer\",\"null\"]") !=
            std::string::npos,
        "Chat Completions converts optional properties to nullable required "
        "properties");
}

} // namespace test

int main() {
  test::test_chat_completions();
  test::test_chat_completions_stream();
  test::test_chat_completions_mistral_content();
  test::test_chat_completions_history_and_image();
  test::test_chat_completions_structured_output();
  return test::failures == 0 ? 0 : 1;
}
