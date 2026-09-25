#pragma once

#include <cail/language_model.hpp>
#include <cail/tool.hpp>

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace cail {

struct GenerateTextOptions {
    LanguageModel model;
    std::optional<std::string> system;
    std::optional<std::string> prompt;
    std::vector<Message> messages;
    std::vector<Tool> tools;
    std::optional<StructuredOutput> structured_output;
    ToolLoopOptions tool_loop;
    std::string session_id;
    std::optional<std::size_t> max_output_tokens;
    std::optional<bool> stream_usage;
};

[[nodiscard]] inline Result<GenerationResponse> generate_text(GenerateTextOptions options)
{
    if (!options.prompt && options.messages.empty()) {
        return std::unexpected(Error{
            .code = ErrorCode::invalid_configuration,
            .message = "Text generation requires a prompt or at least one message.",
        });
    }
    if (options.system) {
        options.messages.insert(options.messages.begin(), Message{
            .role = MessageRole::system,
            .content = {TextPart{.text = std::move(*options.system)}},
        });
    }
    if (options.prompt) {
        options.messages.push_back(Message{
            .role = MessageRole::user,
            .content = {TextPart{.text = std::move(*options.prompt)}},
        });
    }
    GenerationRequest request{
        .messages = std::move(options.messages),
        .structured_output = std::move(options.structured_output),
        .session_id = std::move(options.session_id),
        .max_output_tokens = options.max_output_tokens,
        .stream_usage = options.stream_usage,
    };
    return run_tool_loop(options.model, std::move(request), options.tools, options.tool_loop);
}

template <typename T>
[[nodiscard]] Result<T> generate_object(GenerateTextOptions options)
{
    options.structured_output = StructuredOutput{
        .name = "cail_output",
        .schema = schema<T>(),
    };
    auto response = generate_text(std::move(options));
    if (!response) {
        return std::unexpected(response.error());
    }
    if (response->status == GenerationStatus::refused) {
        return std::unexpected(Error{
            .code = ErrorCode::refused,
            .message = response->text.empty() ? "The model refused the request." : response->text,
        });
    }
    if (response->status == GenerationStatus::incomplete) {
        return std::unexpected(Error{
            .code = ErrorCode::incomplete_response,
            .message = "The model returned an incomplete structured response.",
        });
    }
    if (!response->tool_calls.empty()) {
        return std::unexpected(Error{
            .code = ErrorCode::tool_call_required,
            .message = "The model requested a tool call before returning structured output.",
        });
    }
    return from_json<T>(response->text);
}

} // namespace cail
