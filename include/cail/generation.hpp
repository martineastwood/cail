#pragma once

#include <cail/schema.hpp>

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cail {

// The text view is valid only for the duration of the callback.
using TextDeltaHandler = std::function<void(std::string_view)>;

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
};

struct ToolDefinition {
    std::string name;
    std::string description;
    Schema parameters;
};

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
    std::string content;
    std::string tool_call_id;
    std::vector<ToolCall> tool_calls;
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
};

enum class GenerationStatus {
    completed,
    refused,
    incomplete,
};

struct TokenUsage {
    std::size_t input_tokens{};
    std::size_t output_tokens{};
};

struct ToolResult {
    std::string call_id;
    std::string name;
    std::string output;
};

struct GenerationResponse {
    GenerationStatus status{GenerationStatus::completed};
    std::string text;
    std::optional<TokenUsage> usage;
    std::vector<ToolCall> tool_calls;
    std::vector<ToolResult> tool_results;
    std::optional<std::string> continuation_token;
};

} // namespace cail
