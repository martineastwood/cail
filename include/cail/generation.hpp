#pragma once

#include <cail/schema.hpp>

#include <cstddef>
#include <exception>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace cail {

using ProviderOptions = glz::generic::object_t;

struct HttpRequest;
struct HttpResponse;
struct GenerationResponse;

struct MiddlewareContext {
  std::size_t step{};
};

struct GenerationMiddleware {
  std::function<void(HttpRequest&, const MiddlewareContext&)> before_request;
  std::function<void(const HttpResponse&, const MiddlewareContext&)> after_response;
  std::function<void(const Result<GenerationResponse>&, const MiddlewareContext&)> after_step;
};

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
  ProviderOptions provider_options;
};

struct ToolDefinition {
  std::string name;
  std::string description;
  Schema parameters;
  // JSON object with provider-specific fields for the tool definition.
  ProviderOptions provider_options;
};

struct TextPart {
  std::string text;
  // JSON object with provider-specific fields for this content part.
  ProviderOptions provider_options;
};

struct ImagePart {
  std::string bytes;
  std::string mime_type;
  // JSON object with provider-specific fields for this content part.
  ProviderOptions provider_options;
};

using ContentPart = std::variant<TextPart, ImagePart>;

template <typename Arguments>
[[nodiscard]] ToolDefinition make_tool(std::string name, std::string description) {
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
  ProviderOptions provider_options;
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
  ProviderOptions provider_options;
  std::vector<GenerationMiddleware> middleware;
  std::size_t step{};
};

namespace detail {

[[nodiscard]] inline Result<void> validate_max_output_tokens(const GenerationRequest& request) {
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
  ProviderOptions provider_options;
};

namespace detail {

template <typename Callback>
[[nodiscard]] inline Result<void> run_middleware(const std::vector<GenerationMiddleware>& chain,
                                                 Callback&& callback, std::string_view phase) {
  try {
    for (const auto& middleware : chain) {
      callback(middleware);
    }
    return {};
  } catch (const std::exception& error) {
    return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                 .message = "Generation middleware failed during " +
                                            std::string{phase} + ": " + error.what()});
  } catch (...) {
    return std::unexpected(
        Error{.code = ErrorCode::invalid_configuration,
              .message = "Generation middleware failed during " + std::string{phase} + "."});
  }
}

[[nodiscard]] inline Result<void> run_before_request(const GenerationRequest& request,
                                                     HttpRequest& http) {
  return run_middleware(
      request.middleware,
      [&](const GenerationMiddleware& middleware) {
        if (middleware.before_request) {
          middleware.before_request(http, {.step = request.step});
        }
      },
      "before_request");
}

[[nodiscard]] inline Result<void> run_after_response(const GenerationRequest& request,
                                                     const HttpResponse& response) {
  return run_middleware(
      request.middleware,
      [&](const GenerationMiddleware& middleware) {
        if (middleware.after_response) {
          middleware.after_response(response, {.step = request.step});
        }
      },
      "after_response");
}

[[nodiscard]] inline Result<void> run_after_step(const GenerationRequest& request,
                                                 const Result<GenerationResponse>& response) {
  return run_middleware(
      request.middleware,
      [&](const GenerationMiddleware& middleware) {
        if (middleware.after_step) {
          middleware.after_step(response, {.step = request.step});
        }
      },
      "after_step");
}

} // namespace detail

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

using StreamEvent = std::variant<TextDelta, RefusalDelta, ReasoningDelta, ToolCallArgumentsDelta,
                                 ToolCallReady, UsageUpdate>;
using StreamHandler = std::function<void(const StreamEvent&)>;

} // namespace cail
