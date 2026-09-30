#pragma once

#include <cail/schema.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <functional>
#include <initializer_list>
#include <limits>
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
  ProviderOptions provider_options;
};

struct TextPart {
  std::string text;
  ProviderOptions provider_options;
};

struct ImagePart {
  std::string bytes;
  std::string mime_type;
  ProviderOptions provider_options;
};

struct PdfPart {
  std::string bytes;
  std::string filename;
  ProviderOptions provider_options;
};

using ContentPart = std::variant<TextPart, ImagePart, PdfPart>;

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
  ProviderOptions provider_options;
};

struct StructuredOutput {
  std::string name{"cail_output"};
  std::optional<std::string> description;
  Schema schema;
};

enum class ToolChoiceMode { auto_, none, required, named };

struct ToolChoice {
  ToolChoiceMode mode{ToolChoiceMode::auto_};
  std::string name;
};

struct GenerationRequest {
  std::vector<Message> messages;
  std::vector<ToolDefinition> tools;
  std::optional<StructuredOutput> structured_output;
  // Opaque provider continuation state returned by an earlier response.
  std::optional<std::string> continuation_token;
  // Stable caller-owned conversation ID for providers that route by session.
  std::string session_id;
  std::optional<std::size_t> max_output_tokens;
  std::optional<double> temperature;
  std::optional<double> top_p;
  std::vector<std::string> stop_sequences;
  std::optional<ToolChoice> tool_choice;
  std::optional<bool> stream_usage;
  ProviderOptions provider_options;
  std::vector<GenerationMiddleware> middleware;
  std::size_t step{};
};

namespace detail {

[[nodiscard]] inline GenerationRequest user_prompt_request(std::string_view prompt) {
  return GenerationRequest{
      .messages = {Message{.content = {TextPart{.text = std::string{prompt}}}}},
  };
}

[[nodiscard]] inline Result<void> validate_pdf_parts(const GenerationRequest& request,
                                                     bool supported = true) {
  for (const auto& message : request.messages) {
    for (const auto& part : message.content) {
      if (const auto* pdf = std::get_if<PdfPart>(&part)) {
        if (!supported || message.role != MessageRole::user || pdf->bytes.empty() ||
            pdf->filename.empty()) {
          return std::unexpected(Error{
              .code = ErrorCode::invalid_configuration,
              .message = supported ? "PDF parts require user content, bytes, and a filename."
                                   : "This adapter does not support PDF input.",
          });
        }
      }
    }
  }
  return {};
}

[[nodiscard]] inline Result<void> validate_max_output_tokens(const GenerationRequest& request) {
  if (request.max_output_tokens && *request.max_output_tokens == 0) {
    return std::unexpected(Error{
        .code = ErrorCode::invalid_configuration,
        .message = "max_output_tokens must be greater than zero.",
    });
  }
  return {};
}

[[nodiscard]] inline Result<void> validate_request_controls(const GenerationRequest& request,
                                                            double max_temperature = 2.0,
                                                            std::size_t max_stops = 4) {
  auto invalid = [](std::string message) -> Result<void> {
    return std::unexpected(
        Error{.code = ErrorCode::invalid_configuration, .message = std::move(message)});
  };
  if (request.temperature && (!std::isfinite(*request.temperature) || *request.temperature < 0 ||
                              *request.temperature > max_temperature))
    return invalid("temperature must be finite and between zero and " +
                   std::to_string(max_temperature) + ".");
  if (request.top_p && (!std::isfinite(*request.top_p) || *request.top_p < 0 || *request.top_p > 1))
    return invalid("top_p must be finite and between zero and one.");
  if (request.stop_sequences.size() > max_stops)
    return invalid(max_stops == 0 ? "This adapter does not support stop_sequences."
                                  : "Too many stop_sequences for this adapter.");
  if (std::ranges::any_of(request.stop_sequences, [](const auto& stop) { return stop.empty(); }))
    return invalid("stop_sequences must not contain empty strings.");
  const auto conflict = [&](const ProviderOptions& options, std::string_view key, bool configured) {
    return configured && options.contains(std::string{key});
  };
  const auto& overrides = request.provider_options;
  if (conflict(overrides, "temperature", request.temperature.has_value()) ||
      conflict(overrides, "top_p", request.top_p.has_value()) ||
      conflict(overrides, "stop", !request.stop_sequences.empty()) ||
      conflict(overrides, "stop_sequences", !request.stop_sequences.empty()) ||
      conflict(overrides, "tool_choice", request.tool_choice.has_value()) ||
      conflict(overrides, "max_tokens", request.max_output_tokens.has_value()) ||
      conflict(overrides, "max_output_tokens", request.max_output_tokens.has_value()) ||
      conflict(overrides, "toolConfig", request.tool_choice.has_value()))
    return invalid(
        "Configure each request control once, using either its common field or provider_options.");
  if (const auto found = overrides.find("generationConfig"); found != overrides.end()) {
    if (const auto* config = found->second.get_if<ProviderOptions>();
        config && (conflict(*config, "temperature", request.temperature.has_value()) ||
                   conflict(*config, "topP", request.top_p.has_value()) ||
                   conflict(*config, "stopSequences", !request.stop_sequences.empty()) ||
                   conflict(*config, "maxOutputTokens", request.max_output_tokens.has_value())))
      return invalid("Configure each request control once, using either its common field or "
                     "provider_options.");
  }
  if (request.tool_choice) {
    const auto& choice = *request.tool_choice;
    if (choice.mode != ToolChoiceMode::named && !choice.name.empty())
      return invalid("A tool choice name requires named mode.");
    if ((choice.mode == ToolChoiceMode::required || choice.mode == ToolChoiceMode::named) &&
        request.tools.empty())
      return invalid("Forced tool choice requires tools.");
    if (choice.mode == ToolChoiceMode::named &&
        (choice.name.empty() || !std::ranges::any_of(request.tools, [&](const auto& tool) {
           return tool.name == choice.name;
         })))
      return invalid("Named tool choice must reference a registered tool.");
  }
  return {};
}

enum class ToolChoiceFormat { responses, chat_completions, anthropic };

[[nodiscard]] inline glz::generic encode_tool_choice(const ToolChoice& choice,
                                                     ToolChoiceFormat format) {
  if (format == ToolChoiceFormat::anthropic) {
    if (choice.mode == ToolChoiceMode::named)
      return ProviderOptions{{"type", "tool"}, {"name", choice.name}};
    return ProviderOptions{{"type", choice.mode == ToolChoiceMode::required ? "any"
                                    : choice.mode == ToolChoiceMode::none   ? "none"
                                                                            : "auto"}};
  }
  if (choice.mode == ToolChoiceMode::named) {
    if (format == ToolChoiceFormat::responses)
      return ProviderOptions{{"type", "function"}, {"name", choice.name}};
    return ProviderOptions{{"type", "function"},
                           {"function", ProviderOptions{{"name", choice.name}}}};
  }
  return choice.mode == ToolChoiceMode::required ? "required"
         : choice.mode == ToolChoiceMode::none   ? "none"
                                                 : "auto";
}

} // namespace detail

enum class GenerationStatus {
  completed,
  refused,
  incomplete,
};

enum class FinishReason { stop, length, tool_calls, content_filter, other, unknown };

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

struct GenerationStep {
  std::size_t step{};
  std::string text;
  std::string reasoning;
  FinishReason finish_reason{FinishReason::unknown};
  std::optional<std::string> raw_finish_reason;
  std::optional<TokenUsage> usage;
  std::vector<ToolCall> tool_calls;
};

struct GenerationResponse {
  GenerationStatus status{GenerationStatus::completed};
  std::string text;
  std::string reasoning;
  FinishReason finish_reason{FinishReason::unknown};
  std::optional<std::string> raw_finish_reason;
  // Usage for the final model call.
  std::optional<TokenUsage> usage;
  // Sum of reported usage across steps; absent when no step reports usage.
  std::optional<TokenUsage> total_usage;
  std::vector<GenerationStep> steps;
  std::vector<ToolCall> tool_calls;
  std::vector<ToolResult> tool_results;
  // Messages generated during this call, including every tool round: the
  // assistant messages and tool-result messages that extend a conversation.
  std::vector<Message> turn;
  std::optional<std::string> continuation_token;
  ProviderOptions provider_options;
};

namespace detail {

inline void apply_finish_reason(GenerationResponse& result, const std::optional<std::string>& raw) {
  result.raw_finish_reason = raw;
  const auto matches = [&](std::initializer_list<std::string_view> reasons) {
    return raw && std::ranges::find(reasons, *raw) != reasons.end();
  };
  if (matches({"stop", "end_turn", "stop_sequence", "STOP", "completed"}))
    result.finish_reason = FinishReason::stop;
  else if (matches({"length", "max_tokens", "MAX_TOKENS", "max_output_tokens"}))
    result.finish_reason = FinishReason::length;
  else if (matches({"tool_calls", "tool_use"}))
    result.finish_reason = FinishReason::tool_calls;
  else if (matches({"content_filter", "refusal", "SAFETY", "RECITATION", "PROHIBITED_CONTENT",
                    "BLOCKLIST", "SPII", "IMAGE_SAFETY"}))
    result.finish_reason = FinishReason::content_filter;
  else
    result.finish_reason = !raw || raw->empty() ? FinishReason::unknown : FinishReason::other;
  if (result.status == GenerationStatus::refused)
    result.finish_reason = FinishReason::content_filter;
  else if (result.finish_reason == FinishReason::content_filter)
    result.status = GenerationStatus::refused;
  else if (result.finish_reason == FinishReason::length ||
           result.finish_reason == FinishReason::other)
    result.status = GenerationStatus::incomplete;
}

inline void add_usage(TokenUsage& total, const TokenUsage& usage) {
  total.input_tokens += usage.input_tokens;
  total.output_tokens += usage.output_tokens;
  auto add = [](auto& target, const auto& value) {
    if (value)
      target = target.value_or(0) + *value;
  };
  add(total.cache_read_tokens, usage.cache_read_tokens);
  add(total.cache_write_tokens, usage.cache_write_tokens);
  add(total.reasoning_tokens, usage.reasoning_tokens);
}

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
