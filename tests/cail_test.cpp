#include <cail/cail.hpp>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace test {

int failures{};

void check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

struct Address {
  std::string city;
};

struct Profile {
  std::string answer;
  std::optional<int> confidence;
  std::optional<Address> address;
};

struct ToolInput {
  std::string query;
  std::optional<int> limit;
};

struct ToolOutput {
  int count{};
};

class StubTransport final : public cail::HttpTransport {
public:
  cail::HttpResponse response{
      .status_code = 200,
      .body = R"({"status":"completed","output":[]})",
  };
  std::vector<std::string> chunks;
  cail::HttpRequest request;
  int send_count{};
  int stream_count{};

  [[nodiscard]] cail::Result<cail::HttpResponse>
  send(const cail::HttpRequest &value) override {
    ++send_count;
    request = value;
    return response;
  }

  [[nodiscard]] cail::Result<cail::HttpResponse>
  stream(const cail::HttpRequest &value, const cail::HttpDataHandler &on_data,
         std::stop_token stop) override {
    ++stream_count;
    request = value;
    for (const auto &chunk : chunks) {
      if (stop.stop_requested()) {
        return std::unexpected(cail::Error{.code = cail::ErrorCode::cancelled,
                                           .message = "Cancelled."});
      }
      on_data(chunk);
    }
    if (stop.stop_requested()) {
      return std::unexpected(cail::Error{.code = cail::ErrorCode::cancelled,
                                         .message = "Cancelled."});
    }
    return response;
  }
};

class ScriptedClient {
public:
  explicit ScriptedClient(std::vector<cail::GenerationResponse> responses)
      : responses_(std::move(responses)) {}

  [[nodiscard]] cail::Result<cail::GenerationResponse>
  generate(const cail::GenerationRequest &request) const {
    requests.push_back(request);
    if (responses_.empty()) {
      return std::unexpected(cail::Error{
          .code = cail::ErrorCode::provider_response,
          .message = "The test client ran out of scripted responses.",
      });
    }
    auto response = std::move(responses_.front());
    responses_.erase(responses_.begin());
    return response;
  }

  mutable std::vector<cail::GenerationRequest> requests;

private:
  mutable std::vector<cail::GenerationResponse> responses_;
};

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
  check(confidence.description == "Confidence score" &&
            confidence.minimum == 0.0 && confidence.maximum == 1.0,
        "field assignment preserves schema metadata");
}

void test_optional_schema_and_json() {
  const auto output_schema = cail::schema<Profile>();
  check(output_schema.properties.has_value(),
        "schema exposes model properties");
  check(output_schema.required.has_value(),
        "schema exposes required properties");
  if (!output_schema.properties || !output_schema.required) {
    return;
  }

  check(std::ranges::find(*output_schema.required, "answer") !=
            output_schema.required->end(),
        "non-optional property remains required");
  check(std::ranges::find(*output_schema.required, "confidence") ==
            output_schema.required->end(),
        "optional property stays optional in the generic schema");
  const auto &confidence = *output_schema.properties->at("confidence");
  const auto *confidence_types =
      std::get_if<std::vector<cail::SchemaType>>(&confidence.type);
  check(confidence_types != nullptr,
        "optional property schema uses a type union");
  if (confidence_types != nullptr) {
    check(*confidence_types ==
              std::vector<cail::SchemaType>{cail::SchemaType::integer,
                                            cail::SchemaType::null},
          "optional integer schema allows null");
  }

  auto missing = cail::from_json<Profile>(R"({"answer":"ok"})");
  check(missing.has_value(), "missing optional properties deserialize");
  if (missing) {
    check(!missing->confidence && !missing->address,
          "missing optional values become empty optionals");
  }

  auto null_value = cail::from_json<Profile>(
      R"({"answer":"ok","confidence":null,"address":null})");
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

void test_chat_completions() {
  auto transport = std::make_unique<StubTransport>();
  auto *stub = transport.get();
  stub->response.body =
      R"({"choices":[{"index":0,"finish_reason":"tool_calls","message":{"content":"Searching","tool_calls":[{"id":"call-1","type":"function","function":{"name":"search","arguments":"{\"q\":\"x\"}"}}]}}],"usage":{"prompt_tokens":20,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":10}}})";
  cail::detail::chat_completions::Client client(
      "https://example.test/v1/chat/completions", "test-model", "test-key",
      {{.name = "X-App", .value = "cail"}}, std::move(transport));
  const auto response = client.generate(cail::GenerationRequest{
      .messages = {cail::Message{
          .role = cail::MessageRole::user,
          .content = {cail::TextPart{.text = "Find x"}}}},
      .tools = {cail::make_tool<ToolInput>("search", "Search")},
  });
  check(response && response->text == "Searching" &&
            response->tool_calls.size() == 1 &&
            response->tool_calls[0].name == "search",
        "Chat Completions decodes tool calls");
  check(response && response->usage && response->usage->cache_read_tokens == 10,
        "Chat Completions decodes usage");
  check(stub->request.url == "https://example.test/v1/chat/completions" &&
            stub->request.body.find("\"model\":\"test-model\"") !=
                std::string::npos &&
            stub->request.body.find("\"tools\"") != std::string::npos,
        "Chat Completions sends the configured endpoint, model, and tools");
}

void test_chat_completions_stream() {
  auto transport = std::make_unique<StubTransport>();
  auto *stub = transport.get();
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
  cail::detail::chat_completions::Client client(
      "https://example.test/v1/chat/completions", "test-model", "test-key", {},
      std::move(transport));
  std::vector<cail::StreamEvent> events;
  const auto response = client.stream(
      cail::GenerationRequest{
          .messages = {cail::Message{
              .role = cail::MessageRole::user,
              .content = {cail::TextPart{.text = "Hi"}}}},
      },
      [&](const cail::StreamEvent &event) { events.push_back(event); });
  check(response && response->text == "Hi" &&
            response->tool_calls.size() == 1 &&
            response->tool_calls[0].arguments == R"({"q":"x"})",
        "Chat Completions assembles streamed calls");
  check(response && response->usage && response->usage->input_tokens == 4,
        "Chat Completions receives final streamed usage");
  check(stub->request.body.find("\"include_usage\":true") != std::string::npos,
        "Chat Completions requests streamed usage");
  check(events.size() == 5 &&
            std::get_if<cail::ToolCallArgumentsDelta>(&events[1]) &&
            std::get<cail::ToolCallArgumentsDelta>(events[1]).output_index ==
                1 &&
            std::get_if<cail::ToolCallReady>(&events[4]),
        "Chat Completions preserves tool indexes in events");
}

void test_chat_completions_mistral_content() {
  auto transport = std::make_unique<StubTransport>();
  transport->chunks = {
      "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":[{\"type\":\"thinking\",\"thinking\":[{\"type\":\"text\",\"text\":\"First step\"}]}]}}]}\n\n",
      "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":[{\"type\":\"thinking\",\"thinking\":[{\"type\":\"text\",\"text\":\" next\"}]},{\"type\":\"text\",\"text\":\"Answer\"}]}}]}\n\n",
      "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\" done\"},\"finish_reason\":\"stop\"}]}\n\n",
      "data: [DONE]\n\n",
  };
  cail::detail::chat_completions::Client client(
      "https://example.test/v1/chat/completions", "test-model", "test-key", {},
      std::move(transport));
  std::vector<cail::StreamEvent> events;
  const auto response = client.stream(
      cail::GenerationRequest{.messages = {cail::Message{
          .role = cail::MessageRole::user,
          .content = {cail::TextPart{.text = "Hi"}}}}},
      [&](const cail::StreamEvent &event) { events.push_back(event); });
  check(response && response->reasoning == "First step next" &&
            response->text == "Answer done",
        "Chat Completions decodes Mistral thinking and text parts");
  check(events.size() >= 3 && std::get_if<cail::ReasoningDelta>(&events[0]) &&
            std::get_if<cail::TextDelta>(&events[2]),
        "Chat Completions emits Mistral reasoning before answer text");
}

void test_chat_completions_history_and_image() {
  auto transport = std::make_unique<StubTransport>();
  auto *stub = transport.get();
  stub->response.body =
      R"({"choices":[{"index":0,"finish_reason":"stop","message":{"content":"Done"}}]})";
  cail::detail::chat_completions::Client client(
      "https://example.test/v1/chat/completions", "test-model", "test-key", {},
      std::move(transport));
  const auto
      response =
          client
              .generate(
                  cail::GenerationRequest{
                      .messages =
                          {
                              cail::Message{
                                  .role = cail::MessageRole::user,
                                  .content = {cail::TextPart{.text =
                                                                 "Describe"},
                                              cail::ImagePart{.bytes = std::string{"A\0B", 3}, .mime_type = "image/png"}}},
                              cail::
                                  Message{.role = cail::MessageRole::assistant,
                                          .tool_calls = {{.id = "call-1", .name = "search", .arguments = "{}"}}},
                              cail::Message{.role = cail::MessageRole::tool,
                                            .content = {cail::TextPart{.text = "found"}},
                                            .tool_call_id = "call-1"},
                          },
                  });
  check(response && response->text == "Done",
        "Chat Completions accepts image and tool history");
  check(stub->request.body.find("data:image/png;base64,QQBC") !=
                std::string::npos &&
            stub->request.body.find("\"tool_call_id\":\"call-1\"") !=
                std::string::npos &&
            stub->request.body.find("\"tool_calls\"") != std::string::npos &&
            stub->request.body.find("\"index\"") == std::string::npos,
        "Chat Completions encodes images and matching tool history");
}

void test_chat_completions_structured_output() {
  auto transport = std::make_unique<StubTransport>();
  auto *stub = transport.get();
  stub->response.body =
      R"({"choices":[{"index":0,"finish_reason":"stop","message":{"content":"{\"answer\":\"yes\",\"confidence\":null,\"address\":null}"}}]})";
  cail::detail::chat_completions::Client client(
      "https://example.test/v1/chat/completions", "test-model", "test-key", {},
      std::move(transport));
  const auto response = client.generate(cail::GenerationRequest{
      .messages = {cail::Message{
          .content = {cail::TextPart{.text = "Classify this."}}}},
      .structured_output =
          cail::StructuredOutput{.name = "profile",
                                 .schema = cail::schema<Profile>()},
  });
  check(response &&
            response->text.find(R"("answer":"yes")") != std::string::npos,
        "Chat Completions returns structured JSON text");
  check(stub->request.body.find(
            "\"response_format\":{\"type\":\"json_schema\"") !=
                std::string::npos &&
            stub->request.body.find("\"name\":\"profile\"") !=
                std::string::npos &&
            stub->request.body.find("\"strict\":true") != std::string::npos,
        "Chat Completions sends a strict JSON Schema response format");
  check(stub->request.body.find(
            "\"confidence\":{\"type\":[\"integer\",\"null\"]") !=
            std::string::npos,
        "Chat Completions converts optional properties to nullable required "
        "properties");
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

void test_tool_loop() {
  ScriptedClient client({
      cail::GenerationResponse{
          .tool_calls = {cail::ToolCall{.id = "call-1",
                                        .name = "count",
                                        .arguments = R"({"query":"abc"})"}},
          .continuation_token = "turn-1",
      },
      cail::GenerationResponse{.text = "counted"},
  });
  auto count_tool = cail::tool<ToolInput, ToolOutput>(
      "count", "Count characters",
      [](const ToolInput &input, const cail::ToolContext &context) {
        check(context.call_id == "call-1" && context.round == 0,
              "tool receives call context");
        return ToolOutput{.count = static_cast<int>(input.query.size())};
      });

  const auto result = cail::run_tool_loop(
      client,
      cail::GenerationRequest{
          .messages = {cail::Message{
              .content = {cail::TextPart{.text = "count abc"}}}},
          .session_id = "stable-session",
      },
      std::vector<cail::Tool>{count_tool});
  check(result && result->text == "counted",
        "tool loop returns the final model response");
  check(result && result->tool_results.size() == 1,
        "tool loop records tool output");
  check(result && result->tool_results.front().output == R"({"count":3})",
        "tool output stays typed and serializes");
  check(client.requests.size() == 2, "tool loop makes a follow-up model call");
  check(client.requests.size() == 2 &&
            client.requests[1].continuation_token == "turn-1",
        "tool loop carries continuation state");
  check(client.requests.size() == 2 &&
            client.requests[1].session_id == "stable-session",
        "tool loop carries the caller session ID");
  check(client.requests.size() == 2 &&
            client.requests[1].messages.front().tool_call_id == "call-1",
        "tool loop sends the result for the matching call");
}

void test_foundry_provider() {
  auto transport = std::make_unique<StubTransport>();
  auto *stub = transport.get();
  const cail::FoundryProvider provider = cail::create_foundry({
      .api_key = "test-foundry-key",
  });
  const auto model = provider(cail::FoundryDeployment{
      .endpoint = "https://example.test/openai/"
                  "responses?api-version=2025-04-01-preview",
      .deployment = "deployment-a",
  }, std::move(transport));
  const auto response = model.generate(cail::GenerationRequest{
      .messages = {cail::Message{
          .role = cail::MessageRole::user,
          .content = {cail::TextPart{.text = "Reply with exactly OK."}},
      }},
  });
  check(response.has_value(), "Foundry returns a generation response");
  check(stub->request.url == "https://example.test/openai/"
                             "responses?api-version=2025-04-01-preview",
        "Foundry keeps the deployment Responses URL without appending a path");
  check(stub->request.headers.size() >= 2 &&
            stub->request.headers[0].value == "Bearer test-foundry-key",
        "Foundry authenticates with a Bearer API key");
  check(stub->request.body.find("\"model\":\"deployment-a\"") !=
                std::string::npos &&
            stub->request.body.find("Reply with exactly OK.") !=
                std::string::npos,
        "Foundry requests the deployment model with the prompt");
}

void test_foundry_env_fallback() {
  const char *previous = std::getenv("AZURE_FOUNDRY_API_KEY");
  const std::string saved = previous == nullptr ? "" : previous;
  setenv("AZURE_FOUNDRY_API_KEY", "env-foundry-key", 1);
  auto transport = std::make_unique<StubTransport>();
  auto *stub = transport.get();
  const auto model = cail::create_foundry()(
      cail::FoundryDeployment{
          .endpoint = "https://example.test/openai/responses?api-version=1",
          .deployment = "deployment-b",
      },
      std::move(transport));
  const auto response = model.generate(cail::GenerationRequest{
      .messages = {cail::Message{
          .role = cail::MessageRole::user,
          .content = {cail::TextPart{.text = "hello"}},
      }},
  });
  if (!saved.empty()) {
    setenv("AZURE_FOUNDRY_API_KEY", saved.c_str(), 1);
  } else {
    unsetenv("AZURE_FOUNDRY_API_KEY");
  }
  check(response.has_value(), "Foundry reads the API key from the environment");
  check(stub->request.headers.size() >= 1 &&
            stub->request.headers[0].value == "Bearer env-foundry-key",
        "Foundry falls back to AZURE_FOUNDRY_API_KEY");
}

void test_foundry_missing_endpoint() {
  const auto model = cail::create_foundry_model({
      .api_key = "test-foundry-key",
      .endpoint = "",
      .deployment = "deployment-a",
  });
  const auto response = model.generate(cail::GenerationRequest{
      .messages = {cail::Message{
          .role = cail::MessageRole::user,
          .content = {cail::TextPart{.text = "hello"}},
      }},
  });
  check(!response, "Foundry rejects an empty deployment endpoint");
  check(!response &&
            response.error().code == cail::ErrorCode::invalid_configuration,
        "Foundry maps missing endpoint to invalid configuration");
}

void test_foundry_stream() {
  auto transport = std::make_unique<StubTransport>();
  auto *stub = transport.get();
  stub->chunks = {
      "event: response.output_text.delta\ndata: "
      "{\"type\":\"response.output_text.delta\",\"delta\":\"Hi\"}\n\n",
      "event: response.completed\ndata: "
      "{\"type\":\"response.completed\",\"response\":{\"status\":\"completed\","
      "\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\","
      "\"text\":\"Hi\"}]}]}}\n\n",
  };
  const auto model = cail::create_foundry({
      .api_key = "test-foundry-key",
  })(cail::FoundryDeployment{
      .endpoint = "https://example.test/openai/responses?api-version=1",
      .deployment = "deployment-a",
  }, std::move(transport));
  std::string streamed;
  const auto response = model.stream(
      "hello",
      [&](const cail::StreamEvent &event) {
        if (const auto *delta = std::get_if<cail::TextDelta>(&event)) {
          streamed += delta->text;
        }
      });
  check(response.has_value(), "Foundry streams a generation response");
  check(streamed == "Hi", "Foundry forwards streamed text deltas");
  check(stub->request.headers.size() >= 3 &&
            stub->request.headers[2].value == "text/event-stream",
        "Foundry requests an SSE stream");
}

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
  const auto expected_endpoint = [](OpenCodeService service,
                                    OpenCodeApiFamily family) {
    const std::string root = service == OpenCodeService::zen
                                 ? "https://opencode.ai/zen/v1"
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
  const auto header_value = [](const cail::HttpRequest &request,
                               std::string_view name) {
    for (const auto &header : request.headers) {
      if (header.name == name) return header.value;
    }
    return std::string{};
  };

  for (const auto service : services) {
    for (const auto family : families) {
      auto transport = std::make_unique<StubTransport>();
      auto *stub = transport.get();
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
      const auto &model_location = family == OpenCodeApiFamily::gemini
                                       ? stub->request.url
                                       : stub->request.body;
      check(model_location.find("model-x") != std::string::npos,
            "OpenCode passes the selected model ID through unchanged");
      check(header_value(stub->request, "Authorization") ==
                "Bearer opencode-key",
            "OpenCode authenticates every protocol with a Bearer key");
      check(header_value(stub->request, "x-opencode-session") ==
                "caller-session-17",
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
          "event: response.completed\ndata: {\"response\":{\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"Hi\"}]}]}}\n\n",
      };
    case OpenCodeApiFamily::anthropic_messages:
      return std::vector<std::string>{
          "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n",
          "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
          "data: {\"type\":\"message_stop\"}\n\n",
      };
    case OpenCodeApiFamily::gemini:
      return std::vector<std::string>{
          "data: {\"candidates\":[{\"finishReason\":\"STOP\",\"content\":{\"parts\":[{\"text\":\"Hi\"}]}}]}\n\n",
      };
    }
    std::unreachable();
  };

  for (const auto family : families) {
    auto transport = std::make_unique<StubTransport>();
    auto *stub = transport.get();
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
        [](const cail::StreamEvent &) {});
    check(response.has_value(), "OpenCode streams through the selected protocol adapter");
    check(stub->stream_count == 1,
          "OpenCode sends streaming requests through the streaming transport");
    check(header_value(stub->request, "x-opencode-session") ==
              "caller-stream-session",
          "OpenCode adds the caller session header to streams");
  }

  auto transport = std::make_unique<StubTransport>();
  auto *stub = transport.get();
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
  check(!missing_session &&
            missing_session.error().code == cail::ErrorCode::invalid_configuration,
        "OpenCode rejects a request without a caller session ID");
  check(stub->send_count == 0,
        "OpenCode validates the session ID before sending a request");

  auto helper_transport = std::make_unique<StubTransport>();
  auto *helper_stub = helper_transport.get();
  helper_stub->response.body =
      R"({"choices":[{"index":0,"finish_reason":"stop","message":{"content":"Hello"}}]})";
  const auto helper_model = cail::create_opencode({
      .api_key = "opencode-key",
      .service = OpenCodeService::zen,
  })("model-x", OpenCodeApiFamily::chat_completions,
     std::move(helper_transport));
  const auto helper_response = cail::generate_text({
      .model = helper_model,
      .prompt = "Say hello.",
      .session_id = "caller-helper-session",
  });
  check(helper_response &&
            header_value(helper_stub->request, "x-opencode-session") ==
                "caller-helper-session",
        "generate_text forwards the caller session ID");
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
  test::test_field_value_api();
  test::test_optional_schema_and_json();
  test::test_openai_strict_optional_schemas();
  test::test_openai_reasoning_summary();
  test::test_chat_completions();
  test::test_chat_completions_stream();
  test::test_chat_completions_mistral_content();
  test::test_chat_completions_history_and_image();
  test::test_chat_completions_structured_output();
  test::test_streaming_across_chunk_boundaries();
  test::test_normalized_stream_events();
  test::test_stream_cancellation();
  test::test_tool_loop();
  test::test_http_error_mapping();
  test::test_image_content();
  test::test_foundry_provider();
  test::test_foundry_env_fallback();
  test::test_foundry_missing_endpoint();
  test::test_foundry_stream();
  test::test_opencode_provider();
  return test::failures == 0 ? 0 : 1;
}
