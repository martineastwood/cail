#include "detail/gemini.hpp"
#include "test_support.hpp"
#include <cail/tool.hpp>

#include <cail/gemini.hpp>

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace test {

// StubTransport defaults to an OpenAI-shaped body, so Gemini request tests seed a
// candidate the Gemini decoder accepts.
constexpr std::string_view ok_body =
    R"({"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}]})";

std::string header_value(const cail::HttpRequest& request, std::string_view name) {
  for (const auto& header : request.headers) {
    if (header.name == name) {
      return header.value;
    }
  }
  return {};
}

void test_request_shape() {
  auto transport = std::make_unique<StubTransport>();
  auto* stub = transport.get();
  stub->response.body = ok_body;
  cail::detail::gemini::Client client({.api_key = "test-key", .model = "test-model"},
                                      std::move(transport));

  const auto response = client.generate(cail::detail::user_prompt_request("Hello"));
  check(response.has_value(), "Gemini accepts a single user message");
  check(stub->request.url ==
            "https://generativelanguage.googleapis.com/v1beta/models/test-model:generateContent",
        "Gemini posts to the model generateContent endpoint");
  check(header_value(stub->request, "x-goog-api-key") == "test-key" &&
            header_value(stub->request, "Content-Type") == "application/json" &&
            header_value(stub->request, "Connection") == "close",
        "Gemini sends its API key, content type, and closes the connection");
  check(header_value(stub->request, "Accept").empty(),
        "Gemini omits the event-stream accept header without streaming");

  stub->chunks = {R"(data: {"candidates":[{"content":{"parts":[{"text":"Hi"}]}}]})"
                  "\n\n"};
  const auto streamed =
      client.stream(cail::detail::user_prompt_request("Hello"), [](const cail::StreamEvent&) {});
  check(streamed.has_value(), "Gemini streams a candidate part");
  check(stub->request.url == "https://generativelanguage.googleapis.com/v1beta/models/test-model"
                             ":streamGenerateContent?alt=sse" &&
            header_value(stub->request, "Accept") == "text/event-stream",
        "Gemini streams from the SSE endpoint");
  check(stub->request.body.find(R"("contents":[{"role":"user","parts":[{"text":"Hello"}]}])") !=
            std::string::npos,
        "Gemini encodes a user text message");
}

void test_messages() {
  auto transport = std::make_unique<StubTransport>();
  auto* stub = transport.get();
  stub->response.body = ok_body;
  cail::detail::gemini::Client client({.api_key = "test-key", .model = "test-model"},
                                      std::move(transport));

  const auto response = client.generate(cail::GenerationRequest{
      .messages = {cail::Message{.content = {cail::TextPart{.text = "Hello"}}},
                   cail::Message{.role = cail::MessageRole::assistant,
                                 .content = {cail::TextPart{.text = "Earlier"}}}}});
  check(response.has_value(), "Gemini accepts user and assistant messages");
  check(stub->request.body.find(R"("contents":[{"role":"user","parts":[{"text":"Hello"}]},)"
                                R"({"role":"model","parts":[{"text":"Earlier"}]}])") !=
            std::string::npos,
        "Gemini encodes assistant turns with the model role");
}

void test_system_instruction() {
  auto transport = std::make_unique<StubTransport>();
  auto* stub = transport.get();
  stub->response.body = ok_body;
  cail::detail::gemini::Client client({.api_key = "test-key", .model = "test-model"},
                                      std::move(transport));

  const auto response = client.generate(cail::GenerationRequest{
      .messages = {cail::Message{.role = cail::MessageRole::system,
                                 .content = {cail::TextPart{.text = "Be concise."}}},
                   cail::Message{.content = {cail::TextPart{.text = "Hello"}}}}});
  check(response.has_value(), "Gemini accepts a leading system message");
  check(stub->request.body.find(
            R"("systemInstruction":{"role":"system","parts":[{"text":"Be concise."}]})") !=
            std::string::npos,
        "Gemini hoists the system message into systemInstruction");
  check(stub->request.body.find(R"("contents":[{"role":"user","parts":)") != std::string::npos,
        "Gemini keeps the system message out of contents");
}

void test_tools() {
  auto transport = std::make_unique<StubTransport>();
  auto* stub = transport.get();
  stub->response.body = ok_body;
  cail::detail::gemini::Client client({.api_key = "test-key", .model = "test-model"},
                                      std::move(transport));

  auto request = cail::detail::user_prompt_request("Hello");
  request.tools = {cail::make_tool<ToolInput>("search", "Search for results")};
  const auto response = client.generate(request);
  check(response.has_value(), "Gemini accepts tool declarations");
  check(
      stub->request.body.find(
          R"("tools":[{"functionDeclarations":[{"name":"search","description":"Search for results","parametersJsonSchema":{"type":"object","properties":{"limit":{"type":["integer","null"]},"query":{"type":"string"}},"required":["query"],"additionalProperties":false}}]}])") !=
          std::string::npos,
      "Gemini sends tool declarations as function declarations with a JSON schema");
}

void test_tool_choice() {
  auto transport = std::make_unique<StubTransport>();
  auto* stub = transport.get();
  stub->response.body = ok_body;
  cail::detail::gemini::Client client({.api_key = "test-key", .model = "test-model"},
                                      std::move(transport));
  auto request = cail::detail::user_prompt_request("Hello");
  request.tools = {cail::make_tool<ToolInput>("search", "Search for results")};

  request.tool_choice = cail::ToolChoice{.mode = cail::ToolChoiceMode::required};
  const auto required = client.generate(request);
  check(required.has_value() &&
            stub->request.body.find(R"("toolConfig":{"functionCallingConfig":{"mode":"ANY"}})") !=
                std::string::npos,
        "Gemini maps a required tool choice to ANY mode");

  request.tool_choice = cail::ToolChoice{.mode = cail::ToolChoiceMode::named, .name = "search"};
  const auto named = client.generate(request);
  check(
      named.has_value() &&
          stub->request.body.find(
              R"("toolConfig":{"functionCallingConfig":{"mode":"ANY","allowedFunctionNames":["search"]}})") !=
              std::string::npos,
      "Gemini maps a named tool choice to ANY mode with allowed names");
}

void test_structured_output() {
  auto transport = std::make_unique<StubTransport>();
  auto* stub = transport.get();
  stub->response.body = ok_body;
  cail::detail::gemini::Client client({.api_key = "test-key", .model = "test-model"},
                                      std::move(transport));

  auto request = cail::detail::user_prompt_request("Hello");
  request.structured_output = cail::StructuredOutput{.schema = cail::schema<Profile>()};
  const auto response = client.generate(request);
  check(response.has_value(), "Gemini accepts a structured output schema");
  check(
      stub->request.body.find(
          R"("responseMimeType":"application/json","responseJsonSchema":{"type":"object","properties":)") !=
          std::string::npos,
      "Gemini sends structured output as a JSON response schema");
}

void test_request_controls() {
  auto transport = std::make_unique<StubTransport>();
  auto* stub = transport.get();
  stub->response.body = ok_body;
  cail::detail::gemini::Client client({.api_key = "test-key", .model = "test-model"},
                                      std::move(transport));

  const auto plain = client.generate(cail::detail::user_prompt_request("Hello"));
  check(plain.has_value() && stub->request.body.find("\"generationConfig\"") == std::string::npos,
        "Gemini omits generationConfig without request controls");

  auto request = cail::detail::user_prompt_request("Hello");
  request.max_output_tokens = 128;
  request.temperature = 0.5;
  request.top_p = 0.8;
  request.stop_sequences = {"END"};
  const auto controlled = client.generate(request);
  check(
      controlled.has_value() &&
          stub->request.body.find(
              R"("generationConfig":{"maxOutputTokens":128,"temperature":0.5,"topP":0.8,"stopSequences":["END"]})") !=
              std::string::npos,
      "Gemini maps request controls into generationConfig");
}

void test_attachments() {
  auto transport = std::make_unique<StubTransport>();
  auto* stub = transport.get();
  stub->response.body = ok_body;
  cail::detail::gemini::Client client({.api_key = "test-key", .model = "test-model"},
                                      std::move(transport));

  const auto response = client.generate(cail::GenerationRequest{
      .messages = {cail::Message{
          .content = {cail::TextPart{.text = "Describe"},
                      cail::ImagePart{.bytes = std::string{"A\0B", 3}, .mime_type = "image/png"},
                      cail::PdfPart{.bytes = "PDF", .filename = "paper.pdf"}}}}});
  check(response.has_value(), "Gemini accepts image and PDF parts");
  check(stub->request.body.find(R"("inlineData":{"mimeType":"image/png","data":"QQBC"})") !=
            std::string::npos,
        "Gemini base64-encodes every image byte as inline data");
  check(stub->request.body.find(R"("inlineData":{"mimeType":"application/pdf","data":"UERG"})") !=
            std::string::npos,
        "Gemini sends PDFs as inline application/pdf data");
}

void test_function_call_round_trip() {
  auto transport = std::make_unique<StubTransport>();
  auto* stub = transport.get();
  stub->response.body = ok_body;
  cail::detail::gemini::Client client({.api_key = "test-key", .model = "test-model"},
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
  check(response.has_value(), "Gemini accepts function call history");
  check(stub->request.body.find(
            R"("parts":[{"functionCall":{"name":"search","args":{"q":"x"},"id":"call-1"}}])") !=
            std::string::npos,
        "Gemini replays assistant tool calls as function calls");
  check(
      stub->request.body.find(
          R"({"role":"user","parts":[{"functionResponse":{"name":"search","response":{"output":"found"},"id":"call-1"}}]})") !=
          std::string::npos,
      "Gemini wraps a non-object tool result as its output");

  auto object_result = cail::GenerationRequest{
      .messages = {
          cail::Message{.content = {cail::TextPart{.text = "Find x"}}},
          cail::Message{
              .role = cail::MessageRole::assistant,
              .tool_calls = {{.id = "call-1", .name = "search", .arguments = R"({"q":"x"})"}}},
          cail::Message{.role = cail::MessageRole::tool,
                        .content = {cail::TextPart{.text = R"({"count":3})"}},
                        .tool_call_id = "call-1"},
      }};
  const auto passed = client.generate(object_result);
  check(passed.has_value() &&
            stub->request.body.find(
                R"("functionResponse":{"name":"search","response":{"count":3},"id":"call-1"})") !=
                std::string::npos,
        "Gemini passes a JSON object tool result through unwrapped");
}

void test_rejections() {
  auto transport = std::make_unique<StubTransport>();
  auto* stub = transport.get();
  cail::detail::gemini::Client client({.api_key = "test-key", .model = "test-model"},
                                      std::move(transport));

  const auto developer = client.generate(cail::GenerationRequest{
      .messages = {cail::Message{.role = cail::MessageRole::developer,
                                 .content = {cail::TextPart{.text = "Dev"}}}}});
  check(!developer && developer.error().code == cail::ErrorCode::invalid_configuration,
        "Gemini rejects developer-role messages");

  const auto repeated = client.generate(cail::GenerationRequest{
      .messages = {cail::Message{.role = cail::MessageRole::system,
                                 .content = {cail::TextPart{.text = "One"}}},
                   cail::Message{.role = cail::MessageRole::system,
                                 .content = {cail::TextPart{.text = "Two"}}},
                   cail::Message{.content = {cail::TextPart{.text = "Hello"}}}}});
  check(!repeated && repeated.error().code == cail::ErrorCode::invalid_configuration,
        "Gemini rejects a second system message");

  const auto late = client.generate(cail::GenerationRequest{
      .messages = {cail::Message{.content = {cail::TextPart{.text = "Hello"}}},
                   cail::Message{.role = cail::MessageRole::system,
                                 .content = {cail::TextPart{.text = "Late"}}}}});
  check(!late && late.error().code == cail::ErrorCode::invalid_configuration,
        "Gemini rejects a system message after conversation messages");

  const auto unknown_call = client.generate(cail::GenerationRequest{
      .messages = {cail::Message{.content = {cail::TextPart{.text = "Find x"}}},
                   cail::Message{.role = cail::MessageRole::tool,
                                 .content = {cail::TextPart{.text = "found"}},
                                 .tool_call_id = "missing"}}});
  check(!unknown_call && unknown_call.error().code == cail::ErrorCode::invalid_tool_call,
        "Gemini rejects a tool result with an unknown call ID");

  auto continuation = cail::detail::user_prompt_request("Hello");
  continuation.continuation_token = "cursor";
  const auto continued = client.generate(continuation);
  check(!continued && continued.error().code == cail::ErrorCode::invalid_configuration,
        "Gemini rejects provider continuation state");
}

void test_decode() {
  auto transport = std::make_unique<StubTransport>();
  auto* stub = transport.get();
  cail::detail::gemini::Client client({.api_key = "test-key", .model = "test-model"},
                                      std::move(transport));

  stub->response.body =
      R"({"candidates":[{"content":{"parts":[{"text":"Hello"},{"text":"weighing","thought":true}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":4}})";
  const auto response = client.generate(cail::detail::user_prompt_request("hello"));
  check(response && response->text == "Hello" && response->reasoning == "weighing",
        "Gemini decodes answer text and thought parts");
  check(response && response->finish_reason == cail::FinishReason::stop &&
            response->status == cail::GenerationStatus::completed,
        "Gemini maps STOP to a completed stop");

  stub->response.body =
      R"({"candidates":[{"content":{"parts":[{"text":"partial"}]},"finishReason":"MAX_TOKENS"}]})";
  const auto truncated = client.generate(cail::detail::user_prompt_request("hello"));
  check(truncated && truncated->finish_reason == cail::FinishReason::length &&
            truncated->status == cail::GenerationStatus::incomplete,
        "Gemini maps MAX_TOKENS to an incomplete length stop");

  stub->response.body =
      R"({"candidates":[{"content":{"parts":[{"functionCall":{"name":"search","args":{"q":"x"},"id":"call-1"}}]},"finishReason":"STOP"}]})";
  const auto called = client.generate(cail::detail::user_prompt_request("hello"));
  check(called && called->tool_calls.size() == 1 && called->tool_calls[0].id == "call-1" &&
            called->tool_calls[0].name == "search" &&
            called->tool_calls[0].arguments == R"({"q":"x"})" &&
            called->finish_reason == cail::FinishReason::tool_calls,
        "Gemini decodes function calls into tool calls");

  stub->response.body =
      R"({"candidates":[{"content":{"parts":[{"functionCall":{"name":"search","id":"call-1"}}]}}]})";
  const auto incomplete = client.generate(cail::detail::user_prompt_request("hello"));
  check(!incomplete && incomplete.error().code == cail::ErrorCode::provider_response,
        "Gemini rejects a function call without arguments");
}

void test_usage_arithmetic() {
  auto transport = std::make_unique<StubTransport>();
  transport->response.body =
      R"({"candidates":[{"content":{"parts":[{"text":"Hello"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":4,"cachedContentTokenCount":32,"thoughtsTokenCount":3}})";
  cail::detail::gemini::Client client({.api_key = "test-key", .model = "test-model"},
                                      std::move(transport));

  const auto response = client.generate(cail::detail::user_prompt_request("hello"));
  check(response && response->usage && response->usage->input_tokens == 10 &&
            response->usage->cache_read_tokens == 32 && response->usage->reasoning_tokens == 3,
        "Gemini reports prompt, cached, and thought token counts");
  check(response && response->usage && response->usage->output_tokens == 7,
        "Gemini reports candidate and thought tokens together as output tokens");
}

void test_error_paths() {
  auto transport = std::make_unique<StubTransport>();
  auto* stub = transport.get();
  cail::detail::gemini::Client client({.api_key = "test-key", .model = "test-model"},
                                      std::move(transport));

  stub->response.body = R"({"candidates":[]})";
  const auto empty = client.generate(cail::detail::user_prompt_request("hello"));
  check(!empty && empty.error().code == cail::ErrorCode::provider_response &&
            empty.error().message == "Gemini returned no candidates.",
        "Gemini rejects a response without candidates");

  stub->response.body = R"({"candidates":)";
  const auto malformed = client.generate(cail::detail::user_prompt_request("hello"));
  check(!malformed && malformed.error().code == cail::ErrorCode::provider_response,
        "Gemini rejects a malformed response body");

  stub->response = cail::HttpResponse{
      .status_code = 400,
      .body = R"({"error":{"message":"bad request","status":"INVALID_ARGUMENT"}})"};
  const auto failed = client.generate(cail::detail::user_prompt_request("hello"));
  check(!failed && failed.error().code == cail::ErrorCode::http_status &&
            failed.error().http_status == 400 && failed.error().message == "bad request",
        "Gemini maps an error status to the HTTP error code");
}

void test_streaming() {
  auto transport = std::make_unique<StubTransport>();
  auto* stub = transport.get();
  stub->chunks = {
      "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"weighing\","
      "\"thought\":true},{\"text\":\"Hel\"}]}}]}\n\n"
      "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"lo",
      "\"}]}}]}\n\n",
      "data: {\"candidates\":[{\"finishReason\":\"STOP\"}],\"usageMetadata\":"
      "{\"promptTokenCount\":10,\"candidatesTokenCount\":4,\"thoughtsTokenCount\":3}}\n\n",
  };
  cail::detail::gemini::Client client({.api_key = "test-key", .model = "test-model"},
                                      std::move(transport));

  std::string text;
  std::string reasoning;
  std::vector<cail::StreamEvent> events;
  const auto response = client.stream(
      cail::detail::user_prompt_request("hello"), [&](const cail::StreamEvent& event) {
        events.push_back(event);
        if (const auto* delta = std::get_if<cail::TextDelta>(&event)) {
          text += delta->text;
        } else if (const auto* delta = std::get_if<cail::ReasoningDelta>(&event)) {
          reasoning += delta->text;
        }
      });
  check(text == "Hello", "Gemini reassembles text deltas split across chunks and parts");
  check(reasoning == "weighing", "Gemini emits a reasoning delta for thought parts");
  check(response && response->text == "Hello" && response->reasoning == "weighing" &&
            response->finish_reason == cail::FinishReason::stop,
        "Gemini returns the streamed text, reasoning, and finish reason");
  check(response && response->usage && response->usage->input_tokens == 10 &&
            response->usage->output_tokens == 7,
        "Gemini keeps the usage reported by the stream");
  check(events.size() == 4, "Gemini emits one event per streamed part and update");

  stub->chunks.clear();
  const auto empty =
      client.stream(cail::detail::user_prompt_request("hello"), [](const cail::StreamEvent&) {});
  check(!empty && empty.error().code == cail::ErrorCode::provider_response &&
            empty.error().message == "Gemini returned an empty stream.",
        "Gemini rejects a stream without data events");
}

void test_embeddings() {
  auto transport = std::make_unique<StubTransport>();
  auto* stub = transport.get();
  cail::detail::gemini::EmbeddingClient embeddings({.api_key = "test-key", .model = "test-model"},
                                                   std::optional<std::size_t>{},
                                                   std::move(transport));
  stub->response.body =
      R"({"embeddings":[{"values":[0.5,0.25]},{"values":[0.5,0.25]}],"usageMetadata":{"promptTokenCount":7}})";

  const auto batch = embeddings.embed_many({"one", "two"});
  check(batch && batch->embeddings.size() == 2 && batch->dimensions == 2 &&
            batch->embeddings[0].values == std::vector<float>{0.5F, 0.25F},
        "Gemini decodes batch embeddings with their dimensions");
  check(batch && batch->input_tokens == 7, "Gemini reports prompt tokens as input tokens");
  check(stub->request.url == "https://generativelanguage.googleapis.com/v1beta/models/"
                             "test-model:batchEmbedContents" &&
            stub->request.body.find(
                R"("model":"models/test-model","content":{"parts":[{"text":"one"}]})") !=
                std::string::npos &&
            stub->request.body.find(R"("content":{"parts":[{"text":"two"}]})") !=
                std::string::npos &&
            stub->request.body.find("\"outputDimensionality\"") == std::string::npos,
        "Gemini batches every input as a models-prefixed embedding request");

  auto dimensioned = std::make_unique<StubTransport>();
  auto* dimensioned_stub = dimensioned.get();
  cail::detail::gemini::EmbeddingClient sized({.api_key = "test-key", .model = "models/test-model"},
                                              std::size_t{2}, std::move(dimensioned));
  dimensioned_stub->response.body = R"({"embeddings":[{"values":[0.5,0.25]}]})";
  const auto just_one = sized.embed_many({"one"});
  check(just_one && just_one->dimensions == 2 &&
            dimensioned_stub->request.url.find("/models/models/") == std::string::npos &&
            dimensioned_stub->request.body.find(R"("outputDimensionality":2)") != std::string::npos,
        "Gemini keeps a prefixed model ID single-prefixed and sends dimensions");
}

void test_embedding_decode_errors() {
  auto transport = std::make_unique<StubTransport>();
  auto* stub = transport.get();
  cail::detail::gemini::EmbeddingClient embeddings({.api_key = "test-key", .model = "test-model"},
                                                   std::optional<std::size_t>{},
                                                   std::move(transport));

  stub->response.body = R"({"embeddings":[{"values":[0.5,0.25]}]})";
  const auto short_batch = embeddings.embed_many({"one", "two"});
  check(!short_batch && short_batch.error().code == cail::ErrorCode::provider_response,
        "Gemini rejects an embedding batch that misses an input");

  stub->response.body = R"({"embeddings":[{"values":[]},{"values":[]}]})";
  const auto empty_values = embeddings.embed_many({"one", "two"});
  check(!empty_values && empty_values.error().code == cail::ErrorCode::provider_response,
        "Gemini rejects an empty embedding vector");

  auto dimensioned = std::make_unique<StubTransport>();
  dimensioned->response.body = R"({"embeddings":[{"values":[0.5,0.25,0.125]}]})";
  cail::detail::gemini::EmbeddingClient sized({.api_key = "test-key", .model = "test-model"},
                                              std::size_t{2}, std::move(dimensioned));
  const auto wrong_size = sized.embed_many({"one"});
  check(!wrong_size && wrong_size.error().code == cail::ErrorCode::provider_response,
        "Gemini rejects an embedding whose size differs from the requested dimensions");
}

void test_anonymous_function_call_round_trip() {
  auto transport = std::make_unique<StubTransport>();
  auto* stub = transport.get();
  stub->response.body = ok_body;
  cail::detail::gemini::Client client({.api_key = "test-key", .model = "test-model"},
                                      std::move(transport));

  stub->response.body =
      R"({"candidates":[{"content":{"parts":[{"functionCall":{"name":"search","args":{"q":"x"}}}]},"finishReason":"STOP"}]})";
  const auto decoded = client.generate(cail::detail::user_prompt_request("Find x"));
  check(decoded && decoded->tool_calls.size() == 1 && decoded->tool_calls[0].id == "search-1" &&
            decoded->tool_calls[0].name == "search" &&
            decoded->tool_calls[0].arguments == R"({"q":"x"})" &&
            decoded->finish_reason == cail::FinishReason::tool_calls,
        "Gemini synthesizes search-1 for a function call without an id");

  stub->response.body = ok_body;
  const auto replayed = client.generate(cail::GenerationRequest{
      .messages = {
          cail::Message{.content = {cail::TextPart{.text = "Find x"}}},
          cail::Message{.role = cail::MessageRole::assistant, .tool_calls = decoded->tool_calls},
          cail::Message{.role = cail::MessageRole::tool,
                        .content = {cail::TextPart{.text = "found"}},
                        .tool_call_id = decoded->tool_calls[0].id},
      }});
  check(replayed.has_value(), "Gemini accepts a tool result for a synthesized id");
  check(stub->request.body.find(
            R"("functionCall":{"name":"search","args":{"q":"x"},"id":"search-1"})") !=
            std::string::npos,
        "Gemini replays the synthesized id as the function call id");
  check(stub->request.body.find(
            R"({"name":"search","response":{"output":"found"},"id":"search-1"})") !=
            std::string::npos,
        "Gemini correlates the tool result with the synthesized id");
}

void test_streaming_anonymous_function_call() {
  auto transport = std::make_unique<StubTransport>();
  auto* stub = transport.get();
  stub->chunks = {
      R"(data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"search","args":{"q":"x"}}}]}}]})"
      "\n\n"};
  cail::detail::gemini::Client client({.api_key = "test-key", .model = "test-model"},
                                      std::move(transport));

  std::vector<cail::ToolCallReady> ready;
  const auto response = client.stream(
      cail::detail::user_prompt_request("Find x"), [&](const cail::StreamEvent& event) {
        if (const auto* call = std::get_if<cail::ToolCallReady>(&event)) {
          ready.push_back(*call);
        }
      });
  check(ready.size() == 1 && !ready[0].call.id.empty(),
        "Gemini streams a tool call event with a synthesized id");
  check(response && response->tool_calls.size() == 1 && !ready.empty() &&
            response->tool_calls[0].id == ready[0].call.id &&
            response->tool_calls[0].id == "search-1",
        "Gemini reports the synthesized id on both the stream event and the response");
}

void test_multiple_anonymous_function_calls() {
  auto transport = std::make_unique<StubTransport>();
  auto* stub = transport.get();
  stub->response.body =
      R"({"candidates":[{"content":{"parts":[{"functionCall":{"name":"search","args":{"q":"x"}}},{"functionCall":{"name":"search","args":{"q":"y"}}}]},"finishReason":"STOP"}]})";
  cail::detail::gemini::Client client({.api_key = "test-key", .model = "test-model"},
                                      std::move(transport));

  const auto response = client.generate(cail::detail::user_prompt_request("Find x"));
  check(response && response->tool_calls.size() == 2 && response->tool_calls[0].id == "search-1",
        "Gemini numbers the first anonymous function call from one");
  check(response && response->tool_calls.size() == 2 && response->tool_calls[1].id == "search-2",
        "Gemini numbers each anonymous function call in order");
}

} // namespace test

int main() {
  test::test_request_shape();
  test::test_messages();
  test::test_system_instruction();
  test::test_tools();
  test::test_tool_choice();
  test::test_structured_output();
  test::test_request_controls();
  test::test_attachments();
  test::test_function_call_round_trip();
  test::test_rejections();
  test::test_decode();
  test::test_usage_arithmetic();
  test::test_error_paths();
  test::test_streaming();
  test::test_embeddings();
  test::test_embedding_decode_errors();
  test::test_anonymous_function_call_round_trip();
  test::test_streaming_anonymous_function_call();
  test::test_multiple_anonymous_function_calls();
  return test::failures == 0 ? 0 : 1;
}
