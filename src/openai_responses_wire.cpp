#include "detail/openai_responses_wire.hpp"
#include <algorithm>
#include <cail/detail/base64.hpp>
#include <cail/generation.hpp>
#include <cail/json.hpp>
#include <cail/tool.hpp>
#include <glaze/glaze.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cail::detail::openai::wire {

Result<ResponseItem> decode_response_item(const glz::raw_json& raw, int http_status) {
  ResponseItem item;
  if (const auto error = glz::read<glz::opts{.error_on_unknown_keys = false}>(item, raw.str);
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

Result<GenerationResponse> decode_response(ResponseBody response_body, int http_status) {
  if (response_body.status == "failed") {
    return std::unexpected(Error{
        .code = ErrorCode::provider_response,
        .message = response_body.error ? response_body.error->message
                                       : "OpenAI reported a failed response.",
        .http_status = http_status,
        .provider_code =
            response_body.error && response_body.error->code ? *response_body.error->code : "",
        .provider_type =
            response_body.error && response_body.error->type ? *response_body.error->type : "",
    });
  }

  GenerationResponse result;
  result.continuation_token = response_body.id;
  cail::detail::apply_finish_reason(result, response_body.incomplete_details
                                                ? response_body.incomplete_details->reason
                                                : std::optional<std::string>{response_body.status});
  if (response_body.status == "incomplete") {
    if (result.status != GenerationStatus::refused) {
      result.status = GenerationStatus::incomplete;
    }
  } else if (response_body.status != "completed") {
    return std::unexpected(Error{
        .code = ErrorCode::provider_response,
        .message = "OpenAI returned an unsupported response status: " + response_body.status,
        .http_status = http_status,
    });
  }

  std::vector<glz::raw_json> reasoning_items;
  for (const auto& raw_item : response_body.output) {
    auto decoded_item = decode_response_item(raw_item, http_status);
    if (!decoded_item) {
      return std::unexpected(decoded_item.error());
    }
    const auto& item = *decoded_item;
    if (item.type == "reasoning") {
      for (const auto& part : item.summary) {
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
      if (!item.call_id || item.call_id->empty() || !item.name || item.name->empty() ||
          !item.arguments) {
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
    for (const auto& content : item.content) {
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
    glz::generic details;
    if (const auto error = glz::read_json(details, *encoded); error) {
      return std::unexpected(Error{.code = ErrorCode::json_deserialization,
                                   .message = glz::format_error(error, *encoded)});
    }
    result.provider_options["reasoning_details"] = std::move(details);
  }
  if (response_body.usage) {
    result.usage = TokenUsage{
        .input_tokens = response_body.usage->input_tokens,
        .output_tokens = response_body.usage->output_tokens,
        .cache_read_tokens = response_body.usage->input_tokens_details
                                 ? response_body.usage->input_tokens_details->cached_tokens
                                 : std::nullopt,
        .reasoning_tokens = response_body.usage->output_tokens_details
                                ? response_body.usage->output_tokens_details->reasoning_tokens
                                : std::nullopt,
    };
  }
  if (result.status == GenerationStatus::refused) {
    result.finish_reason = FinishReason::content_filter;
  } else if (result.finish_reason == FinishReason::stop && !result.tool_calls.empty()) {
    result.finish_reason = FinishReason::tool_calls;
  }
  return result;
}

std::string_view role_name(MessageRole role) {
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

Result<std::string> text_content(const Message& message) {
  std::string text;
  for (const auto& part : message.content) {
    if (const auto* value = std::get_if<TextPart>(&part)) {
      text += value->text;
    } else {
      return std::unexpected(Error{
          .code = ErrorCode::invalid_configuration,
          .message = "OpenAI Responses supports image and PDF parts only in user messages.",
      });
    }
  }
  return text;
}

Result<glz::raw_json> input_content(const Message& message) {
  const auto has_image = std::ranges::any_of(message.content, [](const ContentPart& part) {
    return !std::holds_alternative<TextPart>(part);
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
        .message = "OpenAI Responses supports image and PDF parts only in user messages.",
    });
  }

  std::vector<glz::raw_json> parts;
  parts.reserve(message.content.size());
  for (const auto& part : message.content) {
    Result<std::string> encoded;
    if (const auto* value = std::get_if<TextPart>(&part)) {
      if (value->text.empty()) {
        continue;
      }
      encoded = to_json(InputTextPart{.text = value->text});
    } else if (const auto* pdf = std::get_if<PdfPart>(&part)) {
      if (pdf->bytes.empty() || pdf->filename.empty()) {
        return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                     .message = "A PDF part requires bytes and a filename."});
      }
      encoded = to_json(InputFilePart{
          .filename = pdf->filename,
          .file_data = cail::detail::image_data_url("application/pdf", pdf->bytes),
      });
    } else {
      const auto& image = std::get<ImagePart>(part);
      if (image.mime_type.empty() || image.bytes.empty()) {
        return std::unexpected(Error{
            .code = ErrorCode::invalid_configuration,
            .message = "An image part requires non-empty bytes and a MIME type.",
        });
      }
      encoded = to_json(InputImagePart{
          .image_url = cail::detail::image_data_url(image.mime_type, image.bytes),
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
