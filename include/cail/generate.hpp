#pragma once

#include <cail/language_model.hpp>
#include <cail/tool.hpp>

#include <cstddef>
#include <functional>
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
  ProviderOptions provider_options;
  std::vector<GenerationMiddleware> middleware;
};

namespace detail {

[[nodiscard]] inline Result<GenerationRequest>
prepare_generation_request(GenerateTextOptions& options) {
  if (!options.prompt && options.messages.empty()) {
    return std::unexpected(Error{
        .code = ErrorCode::invalid_configuration,
        .message = "Text generation requires a prompt or at least one message.",
    });
  }
  if (options.system) {
    options.messages.insert(options.messages.begin(),
                            Message{
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
  return GenerationRequest{
      .messages = std::move(options.messages),
      .structured_output = std::move(options.structured_output),
      .session_id = std::move(options.session_id),
      .max_output_tokens = options.max_output_tokens,
      .stream_usage = options.stream_usage,
      .provider_options = std::move(options.provider_options),
      .middleware = std::move(options.middleware),
  };
}

} // namespace detail

[[nodiscard]] inline Result<GenerationResponse> generate_text(GenerateTextOptions options) {
  auto request = detail::prepare_generation_request(options);
  if (!request)
    return std::unexpected(request.error());
  return run_tool_loop(options.model, std::move(*request), options.tools, options.tool_loop);
}

[[nodiscard]] inline Result<GenerationResponse> stream_text(GenerateTextOptions options,
                                                            const StreamHandler& on_event) {
  auto request = detail::prepare_generation_request(options);
  if (!request)
    return std::unexpected(request.error());
  if (!on_event) {
    return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                 .message = "Text streaming requires an event handler."});
  }
  return stream_tool_loop(options.model, std::move(*request), options.tools, on_event,
                          options.tool_loop);
}

[[nodiscard]] inline Result<void>
generate_text_async(GenerateTextOptions options, LanguageModel::GenerationCompletion complete) {
  auto request = detail::prepare_generation_request(options);
  if (!request)
    return std::unexpected(request.error());
  return run_tool_loop_async(std::move(options.model), std::move(*request),
                             std::move(options.tools), std::move(complete),
                             std::move(options.tool_loop));
}

namespace detail {

template <typename T>
[[nodiscard]] Result<T> decode_generated_object(Result<GenerationResponse> response) {
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

} // namespace detail

template <typename T> [[nodiscard]] Result<T> generate_object(GenerateTextOptions options) {
  options.structured_output = StructuredOutput{.name = "cail_output", .schema = schema<T>()};
  return detail::decode_generated_object<T>(generate_text(std::move(options)));
}

template <typename T>
[[nodiscard]] Result<void> generate_object_async(GenerateTextOptions options,
                                                 std::function<void(Result<T>)> complete) {
  if (!complete)
    return std::unexpected(
        Error{.code = ErrorCode::invalid_configuration,
              .message = "Async object generation requires a completion handler."});
  options.structured_output = StructuredOutput{.name = "cail_output", .schema = schema<T>()};
  return generate_text_async(std::move(options),
                             [complete = std::move(complete)](Result<GenerationResponse> response) {
                               complete(detail::decode_generated_object<T>(std::move(response)));
                             });
}

} // namespace cail
