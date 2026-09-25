#include <cail/cail.hpp>

#include <chrono>
#include <iostream>
#include <stop_token>
#include <string>
#include <string_view>

struct Answer {
    std::string answer;
};

namespace {

int failures{};

void check(bool condition, std::string_view name)
{
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

cail::GenerationRequest prompt()
{
    return {.messages = {cail::Message{
        .role = cail::MessageRole::user,
        .content = {cail::TextPart{.text = "Hello"}},
    }}};
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 2) return 2;
    const std::string base = argv[1];
    cail::detail::chat_completions::Client chat(base + "/chat", "test-model", "test-key", {});
    const auto chat_result = chat.generate(prompt());
    check(chat_result && chat_result->text == "Hello", "Chat Completions HTTP response");

    std::string chat_deltas;
    const auto chat_stream = chat.stream(prompt(), [&](const cail::StreamEvent& event) {
        if (const auto* delta = std::get_if<cail::TextDelta>(&event)) chat_deltas += delta->text;
    });
    check(chat_stream && chat_stream->text == "Hi" && chat_deltas == "Hi" &&
              chat_stream->usage && chat_stream->usage->input_tokens == 3,
          "Chat Completions SSE and usage");

    cail::detail::chat_completions::Client structured(base + "/chat/structured", "test-model", "test-key", {});
    auto structured_request = prompt();
    structured_request.structured_output = cail::StructuredOutput{
        .name = "answer", .schema = cail::schema<Answer>()};
    const auto structured_response = structured.generate(structured_request);
    check(structured_response && structured_response->text == R"({"answer":"yes"})",
          "Chat Completions sends structured output over HTTP");
    auto unsupported_request = prompt();
    unsupported_request.continuation_token = "response-id";
    const auto unsupported = chat.generate(unsupported_request);
    check(!unsupported && unsupported.error().code == cail::ErrorCode::invalid_configuration,
          "Chat Completions rejects provider continuation state");

    cail::detail::chat_completions::Client truncated_chat(base + "/chat/truncated", "test-model", "test-key", {});
    const auto truncated = truncated_chat.stream(prompt(), [](const cail::StreamEvent&) {});
    check(!truncated && truncated.error().code == cail::ErrorCode::provider_response,
          "Chat Completions rejects an interrupted stream");

    cail::detail::chat_completions::Client bad_chat(base + "/chat/error", "test-model", "test-key", {});
    const auto bad_result = bad_chat.generate(prompt());
    check(!bad_result && bad_result.error().code == cail::ErrorCode::http_status &&
              bad_result.error().http_status == 429 && bad_result.error().provider_code == "rate_limited" &&
              bad_result.error().request_id == "req_local",
          "Chat Completions HTTP error context");

    cail::detail::openai::Client responses({.api_key = "test-key", .model = "test-model", .base_url = base});
    const auto response_result = responses.generate(prompt());
    check(response_result && response_result->text == "Hello", "OpenAI Responses HTTP response");
    std::string response_deltas;
    const auto response_stream = responses.stream(prompt(), [&](const cail::StreamEvent& event) {
        if (const auto* delta = std::get_if<cail::TextDelta>(&event)) response_deltas += delta->text;
    });
    check(response_stream && response_stream->text == "Hi" && response_deltas == "Hi",
          "OpenAI Responses SSE");

    cail::detail::openai::Client truncated_responses(
        {.api_key = "test-key", .model = "test-model", .base_url = base + "/truncated"});
    const auto truncated_response = truncated_responses.stream(prompt(), [](const cail::StreamEvent&) {});
    check(!truncated_response && truncated_response.error().code == cail::ErrorCode::provider_response,
          "OpenAI Responses rejects an interrupted stream");

    cail::detail::anthropic::Client anthropic({
        .api_key = "test-key", .model = "test-model", .base_url = base + "/anthropic"});
    const auto anthropic_result = anthropic.generate(prompt());
    check(anthropic_result && anthropic_result->text == "Searching" &&
              anthropic_result->reasoning == "Need facts" && anthropic_result->tool_calls.size() == 1 &&
              anthropic_result->tool_calls[0].arguments.find("\"q\"") != std::string::npos &&
              anthropic_result->tool_calls[0].arguments.find("\"x\"") != std::string::npos &&
              anthropic_result->usage && anthropic_result->usage->cache_read_tokens == 2 &&
              anthropic_result->usage->cache_write_tokens == 1,
          "Anthropic HTTP maps text, reasoning, tools, and cache usage");
    std::vector<cail::StreamEvent> anthropic_events;
    const auto anthropic_stream = anthropic.stream(prompt(), [&](const cail::StreamEvent& event) {
        anthropic_events.push_back(event);
    });
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

    cail::detail::anthropic::Client bad_anthropic({
        .api_key = "test-key", .model = "test-model", .base_url = base + "/anthropic/error"});
    const auto anthropic_error = bad_anthropic.generate(prompt());
    check(!anthropic_error && anthropic_error.error().code == cail::ErrorCode::http_status &&
              anthropic_error.error().provider_type == "rate_limit_error" &&
              anthropic_error.error().request_id == "anthropic_req_local",
          "Anthropic HTTP error context");
    cail::detail::anthropic::Client truncated_anthropic({
        .api_key = "test-key", .model = "test-model", .base_url = base + "/anthropic/truncated"});
    const auto interrupted_anthropic = truncated_anthropic.stream(prompt(), [](const cail::StreamEvent&) {});
    check(!interrupted_anthropic && interrupted_anthropic.error().code == cail::ErrorCode::provider_response,
          "Anthropic rejects an interrupted stream");

    cail::detail::anthropic::Client validating_anthropic({
        .api_key = "test-key", .model = "test-model", .base_url = base + "/anthropic/validate",
        .max_tokens = 256});
    auto mapping_request = cail::GenerationRequest{
        .messages = {
            cail::Message{.role = cail::MessageRole::system,
                          .content = {cail::TextPart{.text = "Be concise."}}},
            cail::Message{.role = cail::MessageRole::user,
                          .content = {cail::TextPart{.text = "Describe"},
                                      cail::ImagePart{.bytes = std::string{"A\0B", 3}, .mime_type = "image/png"}}},
            cail::Message{.role = cail::MessageRole::assistant,
                          .tool_calls = {{.id = "toolu_1", .name = "search", .arguments = R"({"q":"x"})"}}},
            cail::Message{.role = cail::MessageRole::tool,
                          .content = {cail::TextPart{.text = "found"}}, .tool_call_id = "toolu_1"},
        },
        .tools = {cail::make_tool<Answer>("search", "Search")},
    };
    const auto mapped = validating_anthropic.generate(mapping_request);
    check(mapped && mapped->text == "Done", "Anthropic request maps images, tool history, schema, and headers");
    cail::detail::anthropic::Client structured_anthropic({
        .api_key = "test-key", .model = "test-model", .base_url = base + "/anthropic/structured"});
    auto anthropic_object_request = prompt();
    anthropic_object_request.structured_output = cail::StructuredOutput{
        .name = "answer", .schema = cail::schema<Answer>()};
    const auto anthropic_object = structured_anthropic.generate(anthropic_object_request);
    check(anthropic_object && anthropic_object->text == R"({"answer":"yes"})",
          "Anthropic sends a JSON Schema output format");
    mapping_request.messages[0].role = cail::MessageRole::developer;
    const auto unsupported_anthropic = validating_anthropic.generate(mapping_request);
    check(!unsupported_anthropic && unsupported_anthropic.error().code == cail::ErrorCode::invalid_configuration,
          "Anthropic rejects unsupported developer-role messages");

    cail::detail::gemini::Client gemini({
        .api_key = "test-key", .model = "test-model", .base_url = base + "/gemini"});
    const auto gemini_result = gemini.generate(prompt());
    check(gemini_result && gemini_result->text == "Hello", "Gemini HTTP response");
    std::string gemini_deltas;
    const auto gemini_stream = gemini.stream(prompt(), [&](const cail::StreamEvent& event) {
        if (const auto* delta = std::get_if<cail::TextDelta>(&event)) gemini_deltas += delta->text;
    });
    check(gemini_stream && gemini_stream->text == "Hi" && gemini_deltas == "Hi" &&
              gemini_stream->usage && gemini_stream->usage->input_tokens == 5,
          "Gemini SSE and usage");
    auto gemini_object_request = prompt();
    gemini_object_request.structured_output = cail::StructuredOutput{
        .name = "answer", .schema = cail::schema<Answer>()};
    const auto gemini_object = gemini.generate(gemini_object_request);
    check(gemini_object && gemini_object->text == R"({"answer":"yes"})",
          "Gemini JSON Schema output");
    auto gemini_tool_request = prompt();
    gemini_tool_request.tools = {cail::make_tool<Answer>("search", "Search")};
    const auto gemini_call = gemini.generate(gemini_tool_request);
    check(gemini_call && gemini_call->tool_calls.size() == 1 &&
              gemini_call->tool_calls.front().id.find("signature") != std::string::npos &&
              gemini_call->tool_calls.front().id.find("call_1") != std::string::npos &&
              gemini_call->tool_calls.front().name == "search",
          "Gemini function call and thought signature");
    cail::detail::gemini::Client validating_gemini({
        .api_key = "test-key", .model = "test-model", .base_url = base + "/gemini/validate"});
    cail::GenerationRequest gemini_history{
        .messages = {
            cail::Message{.role = cail::MessageRole::system,
                          .content = {cail::TextPart{.text = "Be concise."}}},
            cail::Message{.role = cail::MessageRole::user,
                          .content = {cail::ImagePart{.bytes = std::string{"A\0B", 3}, .mime_type = "image/png"}}},
            cail::Message{.role = cail::MessageRole::assistant,
                          .tool_calls = {cail::ToolCall{.id = R"({"id":"call_1","signature":"signature"})", .name = "search",
                                                        .arguments = R"({"q":"x"})"}}},
            cail::Message{.role = cail::MessageRole::tool,
                          .content = {cail::TextPart{.text = R"({"answer":"yes"})"}},
                          .tool_call_id = R"({"id":"call_1","signature":"signature"})"},
        },
    };
    const auto gemini_mapped = validating_gemini.generate(gemini_history);
    check(gemini_mapped && gemini_mapped->text == "Done",
          "Gemini maps images, tools, results, and thought signatures");

    cail::detail::anthropic::Client held_anthropic({
        .api_key = "test-key", .model = "test-model", .base_url = base + "/anthropic/hold"});
    std::stop_source anthropic_stop;
    const auto anthropic_began = std::chrono::steady_clock::now();
    const auto cancelled_anthropic = held_anthropic.stream(prompt(), [&](const cail::StreamEvent& event) {
        if (std::holds_alternative<cail::TextDelta>(event)) anthropic_stop.request_stop();
    }, anthropic_stop.get_token());
    check(!cancelled_anthropic && cancelled_anthropic.error().code == cail::ErrorCode::cancelled &&
              std::chrono::steady_clock::now() - anthropic_began < std::chrono::seconds(2),
          "Anthropic cancels the live HTTP stream");

    const auto openai_caps = cail::openai("test-model").capabilities();
    const auto chat_caps = cail::openrouter("test-model").capabilities();
    const auto anthropic_caps = cail::anthropic("test-model").capabilities();
    const auto gemini_caps = cail::gemini("test-model").capabilities();
    check(openai_caps.streaming && openai_caps.image_input && openai_caps.structured_output &&
              openai_caps.continuation && chat_caps.streaming && chat_caps.structured_output &&
              !chat_caps.continuation && anthropic_caps.streaming && anthropic_caps.tools &&
              anthropic_caps.structured_output && gemini_caps.streaming && gemini_caps.tools &&
              gemini_caps.structured_output && !gemini_caps.continuation,
          "Provider capabilities reflect adapter support");

    cail::detail::chat_completions::Client held_chat(base + "/chat/hold", "test-model", "test-key", {});
    std::stop_source stop;
    const auto began = std::chrono::steady_clock::now();
    const auto cancelled = held_chat.stream(prompt(), [&](const cail::StreamEvent& event) {
        if (std::holds_alternative<cail::TextDelta>(event)) stop.request_stop();
    }, stop.get_token());
    check(!cancelled && cancelled.error().code == cail::ErrorCode::cancelled &&
              std::chrono::steady_clock::now() - began < std::chrono::seconds(2),
          "Chat Completions cancels the live HTTP stream");

    return failures == 0 ? 0 : 1;
}
