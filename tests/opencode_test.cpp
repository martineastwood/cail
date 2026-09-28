#include "test_support.hpp"

#include <cail/generate.hpp>
#include <cail/opencode.hpp>

#include <array>

namespace test {

void test_opencode_provider() {
  using cail::OpenCodeApiFamily;
  using cail::OpenCodeService;

  const std::array services{OpenCodeService::zen, OpenCodeService::go};
  const std::array families{
      OpenCodeApiFamily::chat_completions,
      OpenCodeApiFamily::responses,
      OpenCodeApiFamily::anthropic_messages,
      OpenCodeApiFamily::gemini,
  };
  const auto family_response = [](OpenCodeApiFamily family) -> std::string {
    switch (family) {
    case OpenCodeApiFamily::chat_completions:
      return R"({"choices":[{"index":0,"finish_reason":"stop","message":{"content":"Hello"}}]})";
    case OpenCodeApiFamily::responses:
      return R"({"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"Hello"}]}]})";
    case OpenCodeApiFamily::anthropic_messages:
      return R"({"stop_reason":"end_turn","content":[{"type":"text","text":"Hello"}]})";
    case OpenCodeApiFamily::gemini:
      return R"({"candidates":[{"finishReason":"STOP","content":{"parts":[{"text":"Hello"}]}}]})";
    }
    std::unreachable();
  };
  const auto expected_endpoint = [](OpenCodeService service, OpenCodeApiFamily family) {
    const std::string root = service == OpenCodeService::zen ? "https://opencode.ai/zen/v1"
                                                             : "https://opencode.ai/zen/go/v1";
    switch (family) {
    case OpenCodeApiFamily::chat_completions:
      return root + "/chat/completions";
    case OpenCodeApiFamily::responses:
      return root + "/responses";
    case OpenCodeApiFamily::anthropic_messages:
      return root + "/messages";
    case OpenCodeApiFamily::gemini:
      return root + "/models/model-x:generateContent";
    }
    std::unreachable();
  };
  const auto header_value = [](const cail::HttpRequest& request, std::string_view name) {
    for (const auto& header : request.headers) {
      if (header.name == name) {
        return header.value;
      }
    }
    return std::string{};
  };

  for (const auto service : services) {
    for (const auto family : families) {
      auto transport = std::make_unique<StubTransport>();
      auto* stub = transport.get();
      stub->response.body = family_response(family);
      const auto model = cail::create_opencode({
          .api_key = "opencode-key",
          .service = service,
      })("model-x", family, std::move(transport));
      const auto response = model.generate(cail::GenerationRequest{
          .messages = {cail::Message{
              .role = cail::MessageRole::user,
              .content = {cail::TextPart{.text = "Say hello."}},
          }},
          .session_id = "caller-session-17",
      });
      check(response && response->text == "Hello",
            "OpenCode routes through the selected protocol adapter");
      check(stub->request.url == expected_endpoint(service, family),
            "OpenCode selects the endpoint from service and API family");
      const auto& model_location =
          family == OpenCodeApiFamily::gemini ? stub->request.url : stub->request.body;
      check(model_location.find("model-x") != std::string::npos,
            "OpenCode passes the selected model ID through unchanged");
      check(header_value(stub->request, "Authorization") == "Bearer opencode-key",
            "OpenCode authenticates every protocol with a Bearer key");
      check(header_value(stub->request, "x-opencode-session") == "caller-session-17",
            "OpenCode uses the caller session ID in its required header");
      if (family == OpenCodeApiFamily::gemini) {
        check(header_value(stub->request, "x-goog-api-key").empty(),
              "OpenCode Gemini route uses Bearer auth only");
      }
    }
  }

  const auto stream_chunks = [](OpenCodeApiFamily family) {
    switch (family) {
    case OpenCodeApiFamily::chat_completions:
      return std::vector<std::string>{
          "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"}}]}\n\n",
          "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
          "data: [DONE]\n\n",
      };
    case OpenCodeApiFamily::responses:
      return std::vector<std::string>{
          "event: response.output_text.delta\ndata: {\"delta\":\"Hi\"}\n\n",
          "event: response.completed\ndata: "
          "{\"response\":{\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"content\":[{"
          "\"type\":\"output_text\",\"text\":\"Hi\"}]}]}}\n\n",
      };
    case OpenCodeApiFamily::anthropic_messages:
      return std::vector<std::string>{
          "data: "
          "{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\","
          "\"text\":\"Hi\"}}\n\n",
          "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
          "data: {\"type\":\"message_stop\"}\n\n",
      };
    case OpenCodeApiFamily::gemini:
      return std::vector<std::string>{
          "data: "
          "{\"candidates\":[{\"finishReason\":\"STOP\",\"content\":{\"parts\":[{\"text\":\"Hi\"}]}}"
          "]}\n\n",
      };
    }
    std::unreachable();
  };

  for (const auto family : families) {
    auto transport = std::make_unique<StubTransport>();
    auto* stub = transport.get();
    stub->response.body.clear();
    stub->chunks = stream_chunks(family);
    const auto model = cail::create_opencode({
        .api_key = "opencode-key",
        .service = OpenCodeService::go,
    })("model-x", family, std::move(transport));
    const auto response = model.stream(
        cail::GenerationRequest{
            .messages = {cail::Message{
                .role = cail::MessageRole::user,
                .content = {cail::TextPart{.text = "Say hello."}},
            }},
            .session_id = "caller-stream-session",
        },
        [](const cail::StreamEvent&) {});
    check(response.has_value(), "OpenCode streams through the selected protocol adapter");
    check(stub->stream_count == 1,
          "OpenCode sends streaming requests through the streaming transport");
    check(header_value(stub->request, "x-opencode-session") == "caller-stream-session",
          "OpenCode adds the caller session header to streams");
  }

  auto transport = std::make_unique<StubTransport>();
  auto* stub = transport.get();
  const auto model = cail::create_opencode({
      .api_key = "opencode-key",
      .service = OpenCodeService::zen,
  })("model-x", OpenCodeApiFamily::chat_completions, std::move(transport));
  const auto missing_session = model.generate(cail::GenerationRequest{
      .messages = {cail::Message{
          .role = cail::MessageRole::user,
          .content = {cail::TextPart{.text = "Say hello."}},
      }},
  });
  check(!missing_session && missing_session.error().code == cail::ErrorCode::invalid_configuration,
        "OpenCode rejects a request without a caller session ID");
  check(stub->send_count == 0, "OpenCode validates the session ID before sending a request");

  auto helper_transport = std::make_unique<StubTransport>();
  auto* helper_stub = helper_transport.get();
  helper_stub->response.body =
      R"({"choices":[{"index":0,"finish_reason":"stop","message":{"content":"Hello"}}]})";
  const auto helper_model = cail::create_opencode({
      .api_key = "opencode-key",
      .service = OpenCodeService::zen,
  })("model-x", OpenCodeApiFamily::chat_completions, std::move(helper_transport));
  const auto helper_response = cail::generate_text({
      .model = helper_model,
      .prompt = "Say hello.",
      .session_id = "caller-helper-session",
  });
  check(helper_response &&
            header_value(helper_stub->request, "x-opencode-session") == "caller-helper-session",
        "generate_text forwards the caller session ID");
}

} // namespace test

int main() {
  test::test_opencode_provider();
  return test::failures == 0 ? 0 : 1;
}
