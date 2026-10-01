#include "detail/anthropic.hpp"
#include "test_support.hpp"
#include <cail/tool.hpp>

#include <cail/anthropic.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace test {

std::string header_value(const cail::HttpRequest& request, std::string_view name) {
  for (const auto& header : request.headers) {
    if (header.name == name) {
      return header.value;
    }
  }
  return {};
}

std::size_t header_count(const cail::HttpRequest& request, std::string_view name) {
  return static_cast<std::size_t>(std::ranges::count_if(
      request.headers, [name](const cail::HttpHeader& header) { return header.name == name; }));
}

void test_request_shape() {
  auto transport = std::make_unique<StubTransport>();
  auto* stub = transport.get();
  cail::detail::anthropic::Client client({.api_key = "test-key", .model = "test-model"},
                                         std::move(transport));

  const auto response = client.generate(cail::detail::user_prompt_request("Hello"));
  check(response.has_value(), "Anthropic accepts a single user message");
  check(stub->request.url == "https://api.anthropic.com/v1/messages",
        "Anthropic posts messages to /v1/messages");
  check(header_value(stub->request, "x-api-key") == "test-key" &&
            header_value(stub->request, "Authorization") == "Bearer test-key" &&
            header_value(stub->request, "anthropic-version") == "2023-06-01" &&
            header_value(stub->request, "Content-Type") == "application/json",
        "Anthropic sends its API key, bearer token, version, and JSON content type");
  check(header_value(stub->request, "Accept").empty(),
        "Anthropic omits the event-stream accept header without streaming");
  check(stub->request.body.find(R"("model":"test-model","max_tokens":1024)") != std::string::npos,
        "Anthropic sends the model and the configured max_tokens");
  check(stub->request.body.find(
            R"("messages":[{"role":"user","content":[{"type":"text","text":"Hello"}]}])") !=
            std::string::npos,
        "Anthropic encodes a user text message");
}

void test_system_prompt() {
  auto transport = std::make_unique<StubTransport>();
  auto* stub = transport.get();
  cail::detail::anthropic::Client client({.api_key = "test-key", .model = "test-model"},
                                         std::move(transport));

  const auto response = client.generate(cail::GenerationRequest{
      .messages = {cail::Message{.role = cail::MessageRole::system,
                                 .content = {cail::TextPart{.text = "Be concise."}}},
                   cail::Message{.role = cail::MessageRole::user,
                                 .content = {cail::TextPart{.text = "Hello"}}}}});
  check(response.has_value(), "Anthropic accepts a leading system message");
  check(stub->request.body.find(R"("system":[{"type":"text","text":"Be concise."}])") !=
            std::string::npos,
        "Anthropic hoists system text into top-level system blocks");
  check(stub->request.body.find(
            R"("messages":[{"role":"user","content":[{"type":"text","text":"Hello"}]}])") !=
            std::string::npos,
        "Anthropic keeps the system message out of messages");
}

void test_tools_and_tool_choice() {
  auto transport = std::make_unique<StubTransport>();
  auto* stub = transport.get();
  cail::detail::anthropic::Client client({.api_key = "test-key", .model = "test-model"},
                                         std::move(transport));
  const auto with_tools = [] {
    auto request = cail::detail::user_prompt_request("Hello");
    request.tools = {cail::make_tool<ToolInput>("search", "Search for results")};
    return request;
  };

  auto required = with_tools();
  required.tool_choice = cail::ToolChoice{.mode = cail::ToolChoiceMode::required};
  const auto forced = client.generate(required);
  check(forced.has_value(), "Anthropic accepts a required tool choice");
  check(
      stub->request.body.find(
          R"("tools":[{"name":"search","description":"Search for results","input_schema":{"type":"object","properties":{"limit":{"type":["integer","null"]},"query":{"type":"string"}},"required":["query"],"additionalProperties":false}}])") !=
          std::string::npos,
      "Anthropic sends tool declarations with their JSON input schema");
  check(stub->request.body.find(R"("tool_choice":{"type":"any"})") != std::string::npos,
        "Anthropic maps a required tool choice to the any form");

  auto named = with_tools();
  named.tool_choice = cail::ToolChoice{.mode = cail::ToolChoiceMode::named, .name = "search"};
  const auto chosen = client.generate(named);
  check(chosen.has_value() &&
            stub->request.body.find(R"("tool_choice":{"type":"tool","name":"search"})") !=
                std::string::npos,
        "Anthropic maps a named tool choice to the tool form");
}

void test_structured_output() {
  auto transport = std::make_unique<StubTransport>();
  auto* stub = transport.get();
  cail::detail::anthropic::Client client({.api_key = "test-key", .model = "test-model"},
                                         std::move(transport));

  auto request = cail::detail::user_prompt_request("Hello");
  request.structured_output = cail::StructuredOutput{.schema = cail::schema<Profile>()};
  const auto response = client.generate(request);
  check(response.has_value(), "Anthropic accepts a structured output schema");
  check(stub->request.body.find(R"("output_config":{"format":{"type":"json_schema","schema":)") !=
            std::string::npos,
        "Anthropic sends structured output as an output_config JSON schema");
  check(stub->request.body.find(R"("required":["address","answer","confidence"])") !=
            std::string::npos,
        "Anthropic requires optional schema properties in strict mode");
}

void test_attachments() {
  auto transport = std::make_unique<StubTransport>();
  auto* stub = transport.get();
  cail::detail::anthropic::Client client({.api_key = "test-key", .model = "test-model"},
                                         std::move(transport));

  const auto response = client.generate(cail::GenerationRequest{
      .messages = {cail::Message{
          .content = {cail::TextPart{.text = "Describe"},
                      cail::ImagePart{.bytes = std::string{"A\0B", 3}, .mime_type = "image/png"},
                      cail::PdfPart{.bytes = "PDF", .filename = "paper.pdf"}}}}});
  check(response.has_value(), "Anthropic accepts image and PDF parts");
  check(
      stub->request.body.find(
          R"("type":"image","source":{"type":"base64","media_type":"image/png","data":"QQBC"})") !=
          std::string::npos,
      "Anthropic base64-encodes every image byte");
  check(
      stub->request.body.find(
          R"("type":"document","source":{"type":"base64","media_type":"application/pdf","data":"UERG"})") !=
          std::string::npos,
      "Anthropic sends PDFs as base64 documents");
}

void test_tool_round_trip() {
  auto transport = std::make_unique<StubTransport>();
  auto* stub = transport.get();
  cail::detail::anthropic::Client client({.api_key = "test-key", .model = "test-model"},
                                         std::move(transport));

  const auto response = client.generate(cail::GenerationRequest{
      .messages = {
          cail::Message{.content = {cail::TextPart{.text = "Find x"}}},
          cail::Message{
              .role = cail::MessageRole::assistant,
              .tool_calls = {{.id = "call-1", .name = "search", .arguments = R"({"q":"x"})"}}},
          cail::Message{.role = cail::MessageRole::tool,
                        .content = {cail::TextPart{.text = "found"}},
                        .tool_call_id = "call-1"},
      }});
  check(response.has_value(), "Anthropic accepts tool history");
  check(stub->request.body.find(
            R"("type":"tool_use","id":"call-1","name":"search","input":{"q":"x"})") !=
            std::string::npos,
        "Anthropic replays assistant tool calls as tool_use blocks");
  check(
      stub->request.body.find(R"("type":"tool_result","tool_use_id":"call-1","content":"found")") !=
          std::string::npos,
      "Anthropic sends a text-only tool result as a bare JSON string");
}

void test_decode() {
  auto transport = std::make_unique<StubTransport>();
  auto* stub = transport.get();
  cail::detail::anthropic::Client client({.api_key = "test-key", .model = "test-model"},
                                         std::move(transport));

  stub->response.body =
      R"({"content":[{"type":"text","text":"Hello"},{"type":"thinking","thinking":"weighing"}],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":4}})";
  const auto response = client.generate(cail::detail::user_prompt_request("hello"));
  check(response && response->text == "Hello" && response->reasoning == "weighing",
        "Anthropic decodes text and thinking blocks");
  check(response && response->finish_reason == cail::FinishReason::stop &&
            response->status == cail::GenerationStatus::completed,
        "Anthropic maps end_turn to a completed stop");

  stub->response.body =
      R"({"content":[{"type":"text","text":"partial"}],"stop_reason":"max_tokens"})";
  const auto truncated = client.generate(cail::detail::user_prompt_request("hello"));
  check(truncated && truncated->finish_reason == cail::FinishReason::length &&
            truncated->status == cail::GenerationStatus::incomplete,
        "Anthropic maps max_tokens to an incomplete length stop");

  stub->response.body =
      R"({"content":[{"type":"tool_use","id":"call-1","name":"search","input":{"q":"x"}}],"stop_reason":"tool_use"})";
  const auto called = client.generate(cail::detail::user_prompt_request("hello"));
  check(called && called->tool_calls.size() == 1 && called->tool_calls[0].id == "call-1" &&
            called->tool_calls[0].name == "search" &&
            called->tool_calls[0].arguments == R"({"q":"x"})",
        "Anthropic decodes tool_use blocks into tool calls");

  stub->response.body = R"({"content":[{"type":"text","text":"no"}],"stop_reason":"refusal"})";
  const auto refused = client.generate(cail::detail::user_prompt_request("hello"));
  check(refused && refused->finish_reason == cail::FinishReason::content_filter &&
            refused->status == cail::GenerationStatus::refused,
        "Anthropic maps refusal to a refused content filter");
}

void test_usage_arithmetic() {
  auto transport = std::make_unique<StubTransport>();
  transport->response.body =
      R"({"content":[{"type":"text","text":"Hello"}],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":4,"cache_read_input_tokens":32,"cache_creation_input_tokens":8}})";
  cail::detail::anthropic::Client client({.api_key = "test-key", .model = "test-model"},
                                         std::move(transport));

  const auto response = client.generate(cail::detail::user_prompt_request("hello"));
  check(response && response->usage && response->usage->input_tokens == 50,
        "Anthropic reports uncached input plus cache reads and writes as input tokens");
  check(response && response->usage && response->usage->cache_read_tokens == 32 &&
            response->usage->cache_write_tokens == 8 && response->usage->output_tokens == 4,
        "Anthropic preserves cache and output token counts");
}

void test_http_error() {
  auto transport = std::make_unique<StubTransport>();
  transport->response = cail::HttpResponse{
      .status_code = 400,
      .body =
          R"({"type":"error","error":{"type":"invalid_request_error","message":"bad request"}})"};
  cail::detail::anthropic::Client client({.api_key = "test-key", .model = "test-model"},
                                         std::move(transport));

  const auto response = client.generate(cail::detail::user_prompt_request("hello"));
  check(!response && response.error().code == cail::ErrorCode::http_status &&
            response.error().http_status == 400,
        "Anthropic maps an error status to the HTTP error code");
  check(!response && response.error().message == "bad request" &&
            response.error().provider_type == "invalid_request_error",
        "Anthropic retains the provider error message and type");
}

void test_streaming_text() {
  auto transport = std::make_unique<StubTransport>();
  auto* stub = transport.get();
  stub->chunks = {
      "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"content\":[],"
      "\"usage\":{\"input_tokens\":10,\"output_tokens\":1}}}\n\n",
      "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,"
      "\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
      "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,"
      "\"delta\":{\"type\":\"text_delta\",\"text\":\"Hel\"}}\n\n",
      "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,"
      "\"delta\":{\"type\":\"text_delta\",\"tex",
      "t\":\"lo\"}}\n\n",
      "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
      "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":"
      "\"end_turn\"},\"usage\":{\"output_tokens\":4}}\n\n",
      "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
  };
  cail::detail::anthropic::Client client({.api_key = "test-key", .model = "test-model"},
                                         std::move(transport));

  std::vector<std::string> deltas;
  const auto response = client.stream(
      cail::detail::user_prompt_request("hello"), [&](const cail::StreamEvent& event) {
        if (const auto* delta = std::get_if<cail::TextDelta>(&event)) {
          deltas.push_back(delta->text);
        }
      });
  check(deltas == std::vector<std::string>{"Hel", "lo"},
        "Anthropic emits one text delta per event across chunk boundaries");
  check(response && response->text == "Hello" &&
            response->finish_reason == cail::FinishReason::stop,
        "Anthropic returns the reassembled stream text and finish reason");
  check(response && response->usage && response->usage->input_tokens == 10 &&
            response->usage->output_tokens == 4,
        "Anthropic keeps the usage reported across the stream");
  check(stub->request.body.find(R"("stream":true)") != std::string::npos &&
            header_value(stub->request, "Accept") == "text/event-stream",
        "Anthropic requests an event stream when streaming");
}

void test_streaming_tool_call() {
  auto transport = std::make_unique<StubTransport>();
  auto* stub = transport.get();
  stub->chunks = {
      "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,"
      "\"content_block\":{\"type\":\"tool_use\",\"id\":\"call-1\",\"name\":\"search\","
      "\"input\":{}}}\n\n",
      "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,"
      "\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"q\\\":\"}}\n\n",
      "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,"
      "\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"x\\\"}\"}}\n\n",
      "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
      "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":"
      "\"tool_use\"}}\n\n",
      "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
  };
  cail::detail::anthropic::Client client({.api_key = "test-key", .model = "test-model"},
                                         std::move(transport));

  std::vector<cail::ToolCallReady> ready;
  const auto response = client.stream(
      cail::detail::user_prompt_request("search"), [&](const cail::StreamEvent& event) {
        if (const auto* call = std::get_if<cail::ToolCallReady>(&event)) {
          ready.push_back(*call);
        }
      });
  check(ready.size() == 1 && ready[0].output_index == 1 && ready[0].call.id == "call-1" &&
            ready[0].call.name == "search" && ready[0].call.arguments == R"({"q":"x"})",
        "Anthropic assembles streamed tool arguments into a ready call");
  check(response && response->tool_calls.size() == 1 &&
            response->tool_calls[0].arguments == R"({"q":"x"})" &&
            response->finish_reason == cail::FinishReason::tool_calls,
        "Anthropic returns the streamed tool call");
}

void test_caller_authorization_is_preserved() {
  auto transport = std::make_unique<StubTransport>();
  auto* stub = transport.get();
  cail::detail::anthropic::Client client(
      {.api_key = "test-key",
       .model = "test-model",
       .headers = {{.name = "Authorization", .value = "Bearer gateway-token"}}},
      std::move(transport));

  const auto response = client.generate(cail::detail::user_prompt_request("Hello"));
  check(response.has_value(), "Anthropic accepts a caller authorization header");
  check(header_count(stub->request, "Authorization") == 1 &&
            header_value(stub->request, "Authorization") == "Bearer gateway-token",
        "Anthropic keeps a caller authorization header and does not append its own");
  check(header_value(stub->request, "x-api-key") == "test-key",
        "Anthropic still sends its API key alongside a caller authorization header");
}

void test_caller_authorization_is_matched_case_insensitively() {
  auto transport = std::make_unique<StubTransport>();
  auto* stub = transport.get();
  cail::detail::anthropic::Client client(
      {.api_key = "test-key",
       .model = "test-model",
       .headers = {{.name = "authorization", .value = "Bearer gateway-token"}}},
      std::move(transport));

  const auto response = client.generate(cail::detail::user_prompt_request("Hello"));
  check(response.has_value(), "Anthropic accepts a lowercase caller authorization header");
  check(header_count(stub->request, "authorization") == 1 &&
            header_value(stub->request, "authorization") == "Bearer gateway-token" &&
            header_value(stub->request, "Authorization").empty(),
        "Anthropic matches a caller authorization header case-insensitively");
  check(header_value(stub->request, "x-api-key") == "test-key",
        "Anthropic still sends its API key alongside a lowercase caller header");
}

void test_caller_api_key_header_is_preserved() {
  auto transport = std::make_unique<StubTransport>();
  auto* stub = transport.get();
  cail::detail::anthropic::Client client(
      {.api_key = "test-key",
       .model = "test-model",
       .headers = {{.name = "x-api-key", .value = "gateway-key"}}},
      std::move(transport));

  const auto response = client.generate(cail::detail::user_prompt_request("Hello"));
  check(response.has_value(), "Anthropic accepts a caller API key header");
  check(header_count(stub->request, "x-api-key") == 1 &&
            header_value(stub->request, "x-api-key") == "gateway-key",
        "Anthropic keeps a caller x-api-key header and does not append its own");
  check(header_count(stub->request, "Authorization") == 1 &&
            header_value(stub->request, "Authorization") == "Bearer test-key",
        "Anthropic still sends its bearer token alongside a caller API key header");
}

} // namespace test

int main() {
  test::test_request_shape();
  test::test_system_prompt();
  test::test_tools_and_tool_choice();
  test::test_structured_output();
  test::test_attachments();
  test::test_tool_round_trip();
  test::test_decode();
  test::test_usage_arithmetic();
  test::test_http_error();
  test::test_streaming_text();
  test::test_streaming_tool_call();
  test::test_caller_authorization_is_preserved();
  test::test_caller_authorization_is_matched_case_insensitively();
  test::test_caller_api_key_header_is_preserved();
  return test::failures == 0 ? 0 : 1;
}
