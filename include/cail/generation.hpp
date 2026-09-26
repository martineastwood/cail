#pragma once

#include <cail/schema.hpp>

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace cail {

struct HttpRequest;
struct HttpResponse;

enum class MessageRole {
    system,
    developer,
    user,
    assistant,
    tool,
};

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments;
    // Provider-specific round-trip data, such as Gemini thought signatures.
    std::optional<std::string> provider_options;
};

struct ToolDefinition {
    std::string name;
    std::string description;
    Schema parameters;
    // JSON object with provider-specific fields for the tool definition.
    std::optional<std::string> provider_options;
};

struct TextPart {
    std::string text;
    // JSON object with provider-specific fields for this content part.
    std::optional<std::string> provider_options;
};

struct ImagePart {
    std::string bytes;
    std::string mime_type;
    // JSON object with provider-specific fields for this content part.
    std::optional<std::string> provider_options;
};

using ContentPart = std::variant<TextPart, ImagePart>;

template <typename Arguments>
[[nodiscard]] ToolDefinition make_tool(std::string name, std::string description)
{
    return ToolDefinition{
        .name = std::move(name),
        .description = std::move(description),
        .parameters = schema<Arguments>(),
    };
}

struct Message {
    MessageRole role{MessageRole::user};
    std::vector<ContentPart> content;
    std::string tool_call_id;
    std::vector<ToolCall> tool_calls;
    // JSON object with provider-specific fields for this message.
    std::optional<std::string> provider_options;
};

struct StructuredOutput {
    std::string name{"cail_output"};
    std::optional<std::string> description;
    Schema schema;
};

struct GenerationRequest {
    std::vector<Message> messages;
    std::vector<ToolDefinition> tools;
    std::optional<StructuredOutput> structured_output;
    // Opaque provider continuation state returned by an earlier response.
    std::optional<std::string> continuation_token;
    // Stable caller-owned conversation ID for providers that route by session.
    std::string session_id;
    // Optional per-request cap on generated tokens.
    std::optional<std::size_t> max_output_tokens;
    // Request usage details in streamed responses when the provider supports it.
    std::optional<bool> stream_usage;
    // JSON object with provider-specific fields for the generation request.
    std::optional<std::string> provider_options;
    // Runs after the provider request is encoded and before it is sent.
    std::function<void(HttpRequest&)> before_request;
    // Runs after the provider response is received.
    std::function<void(const HttpResponse&)> after_response;
};

namespace detail {

[[nodiscard]] inline Result<void> validate_max_output_tokens(const GenerationRequest& request)
{
    if (request.max_output_tokens && *request.max_output_tokens == 0) {
        return std::unexpected(Error{
            .code = ErrorCode::invalid_configuration,
            .message = "max_output_tokens must be greater than zero.",
        });
    }
    return {};
}

} // namespace detail

enum class GenerationStatus {
    completed,
    refused,
    incomplete,
};

struct TokenUsage {
    std::size_t input_tokens{};
    std::size_t output_tokens{};
    std::optional<std::size_t> cache_read_tokens;
    std::optional<std::size_t> cache_write_tokens;
    std::optional<std::size_t> reasoning_tokens;
};

struct ToolResult {
    std::string call_id;
    std::string name;
    std::string output;
};

struct GenerationResponse {
    GenerationStatus status{GenerationStatus::completed};
    std::string text;
    std::string reasoning;
    std::optional<TokenUsage> usage;
    std::vector<ToolCall> tool_calls;
    std::vector<ToolResult> tool_results;
    std::optional<std::string> continuation_token;
    // JSON object with provider-specific response fields for history round-trips.
    std::optional<std::string> provider_options;
};

struct TextDelta {
    std::string text;
};

struct RefusalDelta {
    std::string text;
};

struct ReasoningDelta {
    std::string text;
};

struct ToolCallArgumentsDelta {
    std::size_t output_index{};
    std::string arguments;
};

struct ToolCallReady {
    std::size_t output_index{};
    ToolCall call;
};

struct UsageUpdate {
    TokenUsage usage;
};

using StreamEvent = std::variant<TextDelta, RefusalDelta, ReasoningDelta, ToolCallArgumentsDelta, ToolCallReady, UsageUpdate>;
using StreamHandler = std::function<void(const StreamEvent&)>;

} // namespace cail
