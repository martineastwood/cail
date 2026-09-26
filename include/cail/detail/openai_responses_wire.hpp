#pragma once

#include <cail/detail/base64.hpp>
#include <cail/generation.hpp>
#include <cail/json.hpp>
#include <cail/tool.hpp>

#include <glaze/glaze.hpp>

#include <algorithm>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cail::detail::openai::wire {

struct InputMessage {
  std::string role;
  glz::raw_json content;
};

struct InputTextPart {
  std::string type{"input_text"};
  std::string text;
};

struct InputImagePart {
  std::string type{"input_image"};
  std::string image_url;
};

struct FunctionCallInput {
  std::string type{"function_call"};
  std::string call_id;
  std::string name;
  std::string arguments;
};

struct FunctionCallOutputInput {
  std::string type{"function_call_output"};
  std::string call_id;
  std::string output;
};

struct FunctionTool {
  std::string type{"function"};
  std::string name;
  std::string description;
  glz::raw_json parameters;
  bool strict{true};
};

struct JsonSchemaFormat {
  std::string type{"json_schema"};
  std::string name;
  std::optional<std::string> description;
  bool strict{true};
  glz::raw_json schema;
};

struct TextOptions {
  JsonSchemaFormat format;
};

struct RequestBody {
  std::string model;
  std::vector<glz::raw_json> input;
  std::optional<std::vector<FunctionTool>> tools;
  std::optional<std::string> previous_response_id;
  std::optional<TextOptions> text;
  std::optional<bool> stream;
  std::optional<std::size_t> max_output_tokens;
};

struct ResponseContent {
  std::string type;
  std::optional<std::string> text;
  std::optional<std::string> refusal;
};

struct ResponseItem {
  std::string type;
  std::vector<ResponseContent> content;
  std::vector<ResponseContent> summary;
  std::optional<std::string> call_id;
  std::optional<std::string> name;
  std::optional<std::string> arguments;
  std::optional<std::string> encrypted_content;
};

struct ResponseUsage {
  std::size_t input_tokens{};
  std::size_t output_tokens{};
  struct InputDetails {
    std::optional<std::size_t> cached_tokens;
  };
  struct OutputDetails {
    std::optional<std::size_t> reasoning_tokens;
  };
  std::optional<InputDetails> input_tokens_details;
  std::optional<OutputDetails> output_tokens_details;
};

struct ProviderError {
  std::string message;
  std::optional<std::string> code;
  std::optional<std::string> type;
};

struct ResponseBody {
  std::optional<std::string> id;
  std::string status;
  std::vector<glz::raw_json> output;
  std::optional<ResponseUsage> usage;
  std::optional<ProviderError> error;
};

struct ErrorBody {
  std::optional<ProviderError> error;
};

struct StreamEventBody {
  std::string type;
  std::optional<std::string> delta;
  std::optional<std::size_t> output_index;
  std::optional<std::string> message;
  std::optional<ResponseBody> response;
  std::optional<ProviderError> error;
};

[[nodiscard]] inline Result<ResponseItem>
decode_response_item(const glz::raw_json &raw, int http_status) {
  ResponseItem item;
  if (const auto error =
          glz::read<glz::opts{.error_on_unknown_keys = false}>(item, raw.str);
      error) {
    return std::unexpected(Error{
        .code = ErrorCode::provider_response,
        .message = glz::format_error(error, raw.str),
        .byte_offset = error.count,
        .http_status = http_status,
    });
  }
  return item;
}

[[nodiscard]] inline Result<GenerationResponse>
decode_response(ResponseBody response_body, int http_status) {
  if (response_body.status == "failed") {
    return std::unexpected(Error{
        .code = ErrorCode::provider_response,
        .message = response_body.error ? response_body.error->message
                                       : "OpenAI reported a failed response.",
        .http_status = http_status,
        .provider_code = response_body.error && response_body.error->code
                             ? *response_body.error->code
                             : "",
        .provider_type = response_body.error && response_body.error->type
                             ? *response_body.error->type
                             : "",
    });
  }

  GenerationResponse result;
  result.continuation_token = response_body.id;
  if (response_body.status == "incomplete") {
    result.status = GenerationStatus::incomplete;
  } else if (response_body.status != "completed") {
    return std::unexpected(Error{
        .code = ErrorCode::provider_response,
        .message = "OpenAI returned an unsupported response status: " +
                   response_body.status,
        .http_status = http_status,
    });
  }

  std::vector<glz::raw_json> reasoning_items;
  for (const auto &raw_item : response_body.output) {
    auto decoded_item = decode_response_item(raw_item, http_status);
    if (!decoded_item) {
      return std::unexpected(decoded_item.error());
    }
    const auto &item = *decoded_item;
    if (item.type == "reasoning") {
      for (const auto &part : item.summary) {
        if (part.text) {
          result.reasoning += *part.text;
        }
      }
      if (item.encrypted_content) {
        reasoning_items.emplace_back(raw_item);
      }
      continue;
    }
    if (item.type == "function_call") {
      if (!item.call_id || item.call_id->empty() || !item.name ||
          item.name->empty() || !item.arguments) {
        return std::unexpected(Error{
            .code = ErrorCode::provider_response,
            .message = "OpenAI returned a function call without its call ID, "
                       "name, or arguments.",
            .http_status = http_status,
        });
      }
      result.tool_calls.push_back(ToolCall{
          .id = *item.call_id,
          .name = *item.name,
          .arguments = *item.arguments,
      });
      continue;
    }
    if (item.type != "message") {
      continue;
    }
    for (const auto &content : item.content) {
      if (content.type == "refusal") {
        result.status = GenerationStatus::refused;
        if (content.refusal) {
          result.text += *content.refusal;
        }
      } else if (content.type == "output_text" && content.text) {
        result.text += *content.text;
      }
    }
  }
  if (!reasoning_items.empty()) {
    auto encoded = to_json(reasoning_items);
    if (!encoded) {
      return std::unexpected(encoded.error());
    }
    result.provider_options = "{\"reasoning_details\":" + *encoded + "}";
  }
  if (response_body.usage) {
    result.usage = TokenUsage{
        .input_tokens = response_body.usage->input_tokens,
        .output_tokens = response_body.usage->output_tokens,
        .cache_read_tokens =
            response_body.usage->input_tokens_details
                ? response_body.usage->input_tokens_details->cached_tokens
                : std::nullopt,
        .reasoning_tokens =
            response_body.usage->output_tokens_details
                ? response_body.usage->output_tokens_details->reasoning_tokens
                : std::nullopt,
    };
  }
  return result;
}

[[nodiscard]] inline std::string_view role_name(MessageRole role) {
  switch (role) {
  case MessageRole::system:
    return "system";
  case MessageRole::developer:
    return "developer";
  case MessageRole::user:
    return "user";
  case MessageRole::assistant:
    return "assistant";
  case MessageRole::tool:
    return {};
  }
  return {};
}

[[nodiscard]] inline Result<std::string> text_content(const Message &message) {
  std::string text;
  for (const auto &part : message.content) {
    if (const auto *value = std::get_if<TextPart>(&part)) {
      text += value->text;
    } else {
      return std::unexpected(Error{
          .code = ErrorCode::invalid_configuration,
          .message =
              "OpenAI Responses supports image parts only in user messages.",
      });
    }
  }
  return text;
}

[[nodiscard]] inline Result<glz::raw_json>
input_content(const Message &message) {
  const auto has_image =
      std::ranges::any_of(message.content, [](const ContentPart &part) {
        return std::holds_alternative<ImagePart>(part);
      });
  if (!has_image) {
    auto value = text_content(message);
    if (!value) {
      return std::unexpected(value.error());
    }
    auto encoded = to_json(*value);
    if (!encoded) {
      return std::unexpected(encoded.error());
    }
    return glz::raw_json{std::move(*encoded)};
  }
  if (message.role != MessageRole::user) {
    return std::unexpected(Error{
        .code = ErrorCode::invalid_configuration,
        .message =
            "OpenAI Responses supports image parts only in user messages.",
    });
  }

  std::vector<glz::raw_json> parts;
  parts.reserve(message.content.size());
  for (const auto &part : message.content) {
    Result<std::string> encoded;
    if (const auto *value = std::get_if<TextPart>(&part)) {
      if (value->text.empty()) {
        continue;
      }
      encoded = to_json(InputTextPart{.text = value->text});
    } else {
      const auto &image = std::get<ImagePart>(part);
      if (image.mime_type.empty() || image.bytes.empty()) {
        return std::unexpected(Error{
            .code = ErrorCode::invalid_configuration,
            .message =
                "An image part requires non-empty bytes and a MIME type.",
        });
      }
      encoded = to_json(InputImagePart{
          .image_url =
              cail::detail::image_data_url(image.mime_type, image.bytes),
      });
    }
    if (!encoded) {
      return std::unexpected(encoded.error());
    }
    parts.emplace_back(std::move(*encoded));
  }
  auto encoded = to_json(parts);
  if (!encoded) {
    return std::unexpected(encoded.error());
  }
  return glz::raw_json{std::move(*encoded)};
}


} // namespace cail::detail::openai::wire
