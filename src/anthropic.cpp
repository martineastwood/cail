#include "detail/anthropic.hpp"
#include "detail/async_http.hpp"
#include "detail/encode_json.hpp"
#include "detail/http_context.hpp"
#include "detail/request_headers.hpp"
#include "detail/sse.hpp"
#include "detail/strict_schema.hpp"
#include <cail/anthropic.hpp>
#include <cail/detail/base64.hpp>
#include <cail/detail/env.hpp>
#include <cail/detail/openai_embeddings.hpp>
#include <cail/embedding_model.hpp>
#include <cail/generation.hpp>
#include <cail/http.hpp>
#include <cail/json.hpp>
#include <cail/language_model.hpp>
#include <cail/schema.hpp>
#include <glaze/glaze.hpp>
#include <map>
#include <memory>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cail::detail::anthropic {

Result<RequestBody> encode(const GenerationRequest& request, const Config& config, bool streaming) {
  if (auto valid = cail::detail::validate_pdf_parts(request); !valid) {
    return std::unexpected(valid.error());
  }
  if (auto valid = cail::detail::validate_request_controls(request, 1.0,
                                                           std::numeric_limits<std::size_t>::max());
      !valid) {
    return std::unexpected(valid.error());
  }
  if (auto valid = cail::detail::validate_max_output_tokens(request); !valid) {
    return std::unexpected(valid.error());
  }
  if (request.messages.empty()) {
    return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                 .message = "Anthropic requires at least one message."});
  }
  if (request.continuation_token) {
    return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                 .message = "Anthropic does not support continuation tokens."});
  }
  RequestBody body{.model = config.model,
                   .max_tokens = request.max_output_tokens.value_or(config.max_tokens)};
  body.temperature = request.temperature;
  body.top_p = request.top_p;
  if (request.tool_choice) {
    body.tool_choice = cail::detail::encode_tool_choice(*request.tool_choice,
                                                        cail::detail::ToolChoiceFormat::anthropic);
  }
  if (!request.stop_sequences.empty()) {
    body.stop_sequences = request.stop_sequences;
  }
  if (streaming) {
    body.stream = true;
  }
  if (request.structured_output) {
    auto schema = cail::detail::strict_json_schema(request.structured_output->schema);
    if (!schema) {
      return std::unexpected(schema.error());
    }
    auto encoded = to_json(*schema);
    if (!encoded) {
      return std::unexpected(encoded.error());
    }
    body.output_config = RequestBody::OutputConfig{
        .format =
            RequestBody::OutputConfig::Format{
                .schema = glz::raw_json{std::move(*encoded)},
            },
    };
  }
  for (const auto& message : request.messages) {
    if (message.role == MessageRole::system) {
      if (!body.messages.empty() || !message.tool_calls.empty() || !message.tool_call_id.empty()) {
        return std::unexpected(
            Error{.code = ErrorCode::invalid_configuration,
                  .message = "Anthropic system messages must precede conversation messages."});
      }
      for (const auto& part : message.content) {
        const auto* text = std::get_if<cail::TextPart>(&part);
        if (!text) {
          return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                       .message = "Anthropic system content must be text."});
        }
        if (body.system) {
          body.system->append("\n");
        } else {
          body.system.emplace();
        }
        body.system->append(text->text);
      }
      continue;
    }
    if (message.role == MessageRole::developer) {
      return std::unexpected(
          Error{.code = ErrorCode::invalid_configuration,
                .message = "Anthropic does not accept developer-role messages."});
    }
    InputMessage item{.role = message.role == MessageRole::assistant ? "assistant" : "user"};
    if (message.role == MessageRole::tool) {
      if (message.tool_call_id.empty() || !message.tool_calls.empty()) {
        return std::unexpected(Error{
            .code = ErrorCode::invalid_tool_call,
            .message = "Anthropic tool results require a call ID and cannot contain tool calls."});
      }
      std::string text;
      bool has_image = false;
      for (const auto& part : message.content) {
        if (const auto* value = std::get_if<cail::TextPart>(&part)) {
          text += value->text;
        } else {
          has_image = true;
        }
      }
      Result<std::string> encoded_content;
      if (!has_image) {
        encoded_content = to_json(text);
      } else {
        std::vector<glz::raw_json> content;
        if (!text.empty()) {
          if (auto added = append_json(content, TextBlock{.text = text}); !added) {
            return std::unexpected(added.error());
          }
        }
        for (const auto& part : message.content) {
          const auto* image = std::get_if<cail::ImagePart>(&part);
          if (!image) {
            continue;
          }
          if (image->mime_type.empty() || image->bytes.empty()) {
            return std::unexpected(
                Error{.code = ErrorCode::invalid_configuration,
                      .message = "Anthropic tool result images require bytes and a MIME type."});
          }
          if (auto added = append_json(
                  content, ImageBlock{.source = ImageSource{.media_type = image->mime_type,
                                                            .data = cail::detail::base64_encode(
                                                                image->bytes)}});
              !added) {
            return std::unexpected(added.error());
          }
        }
        encoded_content = to_json(content);
      }
      if (!encoded_content) {
        return std::unexpected(encoded_content.error());
      }
      auto added = append_json(
          item.content, ToolResultBlock{.tool_use_id = message.tool_call_id,
                                        .content = glz::raw_json{std::move(*encoded_content)}});
      if (!added) {
        return std::unexpected(added.error());
      }
    } else {
      if (!message.tool_call_id.empty() ||
          (message.role != MessageRole::assistant && !message.tool_calls.empty())) {
        return std::unexpected(
            Error{.code = ErrorCode::invalid_tool_call,
                  .message = "Anthropic found a misplaced tool call ID or tool call."});
      }
      for (const auto& part : message.content) {
        Result<void> added;
        if (const auto* text = std::get_if<cail::TextPart>(&part)) {
          if (text->text.empty()) {
            continue;
          }
          added = append_json(item.content, TextBlock{.text = text->text});
        } else if (const auto* pdf = std::get_if<cail::PdfPart>(&part)) {
          added = append_json(
              item.content,
              ImageBlock{.type = "document",
                         .source = ImageSource{.media_type = "application/pdf",
                                               .data = cail::detail::base64_encode(pdf->bytes)}});
        } else {
          const auto& image = std::get<cail::ImagePart>(part);
          if (message.role != MessageRole::user || image.mime_type.empty() || image.bytes.empty()) {
            return std::unexpected(
                Error{.code = ErrorCode::invalid_configuration,
                      .message = "Anthropic images require user content, bytes, and a MIME type."});
          }
          added = append_json(
              item.content,
              ImageBlock{.source = ImageSource{.media_type = image.mime_type,
                                               .data = cail::detail::base64_encode(image.bytes)}});
        }
        if (!added) {
          return std::unexpected(added.error());
        }
      }
      for (const auto& call : message.tool_calls) {
        if (call.id.empty() || call.name.empty() || call.arguments.empty() ||
            glz::validate_json(call.arguments)) {
          return std::unexpected(
              Error{.code = ErrorCode::invalid_tool_call,
                    .message =
                        "Anthropic assistant tool calls require an ID, name, and JSON arguments."});
        }
        auto added = append_json(
            item.content,
            ToolUseBlock{.id = call.id, .name = call.name, .input = glz::raw_json{call.arguments}});
        if (!added) {
          return std::unexpected(added.error());
        }
      }
    }
    body.messages.push_back(std::move(item));
  }
  if (body.messages.empty()) {
    return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                 .message = "Anthropic requires a user or assistant message."});
  }
  if (!request.tools.empty()) {
    body.tools.emplace();
    for (const auto& tool : request.tools) {
      if (tool.name.empty()) {
        return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                     .message = "Tool names cannot be empty."});
      }
      auto schema = to_json(tool.parameters);
      if (!schema) {
        return std::unexpected(schema.error());
      }
      body.tools->push_back(Tool{.name = tool.name,
                                 .description = tool.description,
                                 .input_schema = glz::raw_json{std::move(*schema)}});
    }
  }
  return body;
}

void apply_usage(TokenUsage& target, const Usage& usage) {
  const auto uncached_input =
      usage.input_tokens.value_or(target.input_tokens - target.cache_read_tokens.value_or(0) -
                                  target.cache_write_tokens.value_or(0));
  if (usage.output_tokens) {
    target.output_tokens = *usage.output_tokens;
  }
  if (usage.cache_read_input_tokens) {
    target.cache_read_tokens = usage.cache_read_input_tokens;
  }
  if (usage.cache_creation_input_tokens) {
    target.cache_write_tokens = usage.cache_creation_input_tokens;
  }
  target.input_tokens =
      uncached_input + target.cache_read_tokens.value_or(0) + target.cache_write_tokens.value_or(0);
}

Result<void> retain_thinking(GenerationResponse& result, const std::vector<ThinkingBlock>& blocks) {
  if (blocks.empty()) {
    return {};
  }
  auto encoded = to_json(blocks);
  if (!encoded) {
    return std::unexpected(encoded.error());
  }
  glz::generic details;
  if (const auto error = glz::read_json(details, *encoded); error) {
    return std::unexpected(Error{.code = ErrorCode::json_deserialization,
                                 .message = glz::format_error(error, *encoded)});
  }
  result.provider_options["reasoning_details"] = std::move(details);
  return {};
}

Result<GenerationResponse> decode(const ResponseBody& body) {
  GenerationResponse result;
  cail::detail::apply_finish_reason(result, body.stop_reason);
  std::vector<ThinkingBlock> thinking;
  for (const auto& block : body.content) {
    if (block.type == "thinking" || block.type == "redacted_thinking") {
      thinking.push_back({.type = block.type,
                          .thinking = block.thinking,
                          .signature = block.signature,
                          .data = block.data});
    }
    if (block.type == "text" && block.text) {
      result.text += *block.text;
    } else if (block.type == "thinking" && block.thinking) {
      result.reasoning += *block.thinking;
    } else if (block.type == "tool_use") {
      if (!block.id || !block.name || block.input.str.empty()) {
        return std::unexpected(Error{.code = ErrorCode::provider_response,
                                     .message = "Anthropic returned an incomplete tool call."});
      }
      result.tool_calls.push_back(
          cail::ToolCall{.id = *block.id, .name = *block.name, .arguments = block.input.str});
    }
  }
  if (auto retained = retain_thinking(result, thinking); !retained) {
    return std::unexpected(retained.error());
  }
  if (body.usage) {
    result.usage.emplace();
    apply_usage(*result.usage, *body.usage);
  }
  return result;
}

Result<std::string> apply_message_options(std::string_view encoded,
                                          const GenerationRequest& request) {
  glz::generic body;
  if (const auto error = glz::read_json(body, encoded); error) {
    return std::unexpected(Error{.code = ErrorCode::json_deserialization,
                                 .message = glz::format_error(error, encoded)});
  }
  const auto cache = [](glz::generic& target, const ProviderOptions& options) -> Result<void> {
    if (options.empty()) {
      return {};
    }
    if (options.contains("cache_control")) {
      target["cache_control"] = options.at("cache_control");
    }
    return {};
  };
  glz::generic::array_t system;
  std::size_t index = 0;
  for (const auto& message : request.messages) {
    if (message.role == MessageRole::system) {
      for (const auto& part : message.content) {
        const auto& text = std::get<cail::TextPart>(part);
        if (text.text.empty()) {
          continue;
        }
        glz::generic block = glz::generic::object_t{};
        block["type"] = "text";
        block["text"] = text.text;
        if (auto result = cache(block, text.provider_options); !result) {
          return std::unexpected(result.error());
        }
        system.push_back(std::move(block));
      }
      continue;
    }
    auto& parts = body["messages"]
                      .get<glz::generic::array_t>()[index++]["content"]
                      .get<glz::generic::array_t>();
    if (message.role == MessageRole::tool) {
      if (auto result = cache(parts.front(), message.provider_options); !result) {
        return std::unexpected(result.error());
      }
    } else {
      std::size_t part_index = 0;
      for (const auto& part : message.content) {
        if (const auto* text = std::get_if<cail::TextPart>(&part); text && text->text.empty()) {
          continue;
        }
        auto result = std::visit(
            [&](const auto& value) { return cache(parts[part_index], value.provider_options); },
            part);
        if (!result) {
          return std::unexpected(result.error());
        }
        ++part_index;
      }
      if (message.role == MessageRole::assistant &&
          message.provider_options.contains("reasoning_details")) {
        if (const auto* details =
                message.provider_options.at("reasoning_details").get_if<glz::generic::array_t>()) {
          glz::generic::array_t thinking;
          for (const auto& block : *details) {
            if (!block.is_object() || !block.contains("type")) {
              continue;
            }
            const auto* type = block["type"].get_if<std::string>();
            if (type && (*type == "thinking" || *type == "redacted_thinking")) {
              thinking.push_back(block);
            }
          }
          parts.insert(parts.begin(), thinking.begin(), thinking.end());
        }
      }
    }
  }
  if (!system.empty()) {
    body["system"] = std::move(system);
  }
  if (!request.tools.empty()) {
    auto& tools = body["tools"].get<glz::generic::array_t>();
    for (std::size_t i = 0; i < tools.size(); ++i) {
      if (auto result = cache(tools[i], request.tools[i].provider_options); !result) {
        return std::unexpected(result.error());
      }
    }
  }
  auto result = body.dump();
  if (!result) {
    return std::unexpected(Error{.code = ErrorCode::json_serialization,
                                 .message = "Could not encode Anthropic message options."});
  }
  return std::move(*result);
}

} // namespace cail::detail::anthropic

namespace cail::detail::anthropic {

struct Client::StreamState {
  StreamHandler on_event;
  std::stop_token stop;
  cail::detail::SseParser parser;
  GenerationResponse partial;
  std::map<std::size_t, cail::ToolCall> pending_calls;
  std::map<std::size_t, ThinkingBlock> thinking_blocks;
  bool finished = false;
  std::optional<Error> stream_error;
  void handle_event(const cail::detail::ServerSentEvent& event) {
    if (stop.stop_requested() || stream_error || event.data.empty()) {
      return;
    }
    StreamBody chunk;
    if (const auto error = glz::read<glz::opts{.error_on_unknown_keys = false}>(chunk, event.data);
        error) {
      stream_error = Error{.code = ErrorCode::provider_response,
                           .message = glz::format_error(error, event.data)};
      return;
    }
    if (chunk.type == "error") {
      stream_error =
          Error{.code = ErrorCode::provider_response,
                .message = chunk.error ? chunk.error->message : "Anthropic stream failed.",
                .provider_type = chunk.error ? chunk.error->type : ""};
    } else if (chunk.type == "message_start" && chunk.message && chunk.message->usage) {
      partial.usage.emplace();
      apply_usage(*partial.usage, *chunk.message->usage);
      on_event(StreamEvent{UsageUpdate{.usage = *partial.usage}});
    } else if (chunk.type == "content_block_start" && chunk.index && chunk.content_block &&
               chunk.content_block->type == "tool_use") {
      auto& call = pending_calls[*chunk.index];
      call.id = chunk.content_block->id.value_or("");
      call.name = chunk.content_block->name.value_or("");
      call.arguments = chunk.content_block->input.str == "{}" ? "" : chunk.content_block->input.str;
    } else if (chunk.type == "content_block_start" && chunk.index && chunk.content_block &&
               (chunk.content_block->type == "thinking" ||
                chunk.content_block->type == "redacted_thinking")) {
      const auto& block = *chunk.content_block;
      thinking_blocks[*chunk.index] = {.type = block.type,
                                       .thinking = block.thinking,
                                       .signature = block.signature,
                                       .data = block.data};
      if (block.thinking && !block.thinking->empty()) {
        partial.reasoning += *block.thinking;
        on_event(StreamEvent{ReasoningDelta{.text = *block.thinking}});
      }
    } else if (chunk.type == "content_block_delta" && chunk.index && chunk.delta) {
      const auto& delta = *chunk.delta;
      if (delta.type == "text_delta" && delta.text) {
        partial.text += *delta.text;
        on_event(StreamEvent{TextDelta{.text = *delta.text}});
      } else if (delta.type == "thinking_delta" && delta.thinking) {
        auto& block = thinking_blocks[*chunk.index];
        block.type = "thinking";
        if (!block.thinking) {
          block.thinking.emplace();
        }
        *block.thinking += *delta.thinking;
        partial.reasoning += *delta.thinking;
        on_event(StreamEvent{ReasoningDelta{.text = *delta.thinking}});
      } else if (delta.type == "signature_delta" && delta.signature) {
        auto& signature = thinking_blocks[*chunk.index].signature;
        if (!signature) {
          signature.emplace();
        }
        *signature += *delta.signature;
      } else if (delta.type == "input_json_delta" && delta.partial_json) {
        pending_calls[*chunk.index].arguments += *delta.partial_json;
        on_event(StreamEvent{ToolCallArgumentsDelta{.output_index = *chunk.index,
                                                    .arguments = *delta.partial_json}});
      }
    } else if (chunk.type == "content_block_stop" && chunk.index) {
      if (auto found = pending_calls.find(*chunk.index); found != pending_calls.end()) {
        if (found->second.arguments.empty()) {
          found->second.arguments = "{}";
        }
        if (found->second.id.empty() || found->second.name.empty() ||
            glz::validate_json(found->second.arguments)) {
          stream_error = Error{.code = ErrorCode::provider_response,
                               .message = "Anthropic returned an incomplete tool call."};
          return;
        }
        partial.tool_calls.push_back(found->second);
        on_event(StreamEvent{ToolCallReady{.output_index = *chunk.index, .call = found->second}});
        pending_calls.erase(found);
      }
    } else if (chunk.type == "message_delta") {
      if (chunk.delta) {
        if (chunk.delta->stop_reason) {
          cail::detail::apply_finish_reason(partial, chunk.delta->stop_reason);
        }
      }
      if (chunk.usage) {
        if (!partial.usage) {
          partial.usage.emplace();
        }
        apply_usage(*partial.usage, *chunk.usage);
        on_event(StreamEvent{UsageUpdate{.usage = *partial.usage}});
      }
    } else if (chunk.type == "message_stop") {
      finished = true;
    }
  }
  void feed(std::string_view bytes) {
    parser.feed(bytes, [this](const auto& event) { handle_event(event); });
  }
  Result<GenerationResponse> finish(const HttpResponse& response) {
    if (stop.stop_requested()) {
      return std::unexpected(generation_cancelled_error());
    }
    const auto context = [&](Error error) {
      return unexpected_with_http_context<GenerationResponse>(std::move(error), response);
    };
    if (is_http_error_status(response.status_code)) {
      return context(http_status_error_from_json_body(response));
    }
    parser.finish([this](const auto& event) { handle_event(event); });
    if (stream_error) {
      return context(*stream_error);
    }
    if (!finished || !pending_calls.empty()) {
      return context(Error{
          .code = ErrorCode::provider_response,
          .message = "Anthropic closed the stream before message_stop or a completed tool call."});
    }
    std::vector<ThinkingBlock> thinking;
    for (const auto& [index, block] : thinking_blocks) {
      thinking.push_back(block);
    }
    if (auto retained = retain_thinking(partial, thinking); !retained) {
      return context(retained.error());
    }
    return partial;
  }
};

Client::Client(Config config, std::unique_ptr<HttpTransport> transport)
    : config_(std::move(config)), transport_(std::move(transport)) {}

Result<GenerationResponse> Client::generate(const GenerationRequest& request,
                                            const std::stop_token& stop) const {
  return run(request, {}, stop);
}

Result<GenerationResponse> Client::stream(const GenerationRequest& request,
                                          const StreamHandler& on_event,
                                          const std::stop_token& stop) const {
  return run(request, on_event, stop);
}

Result<void> Client::generate_async(GenerationRequest request,
                                    LanguageModel::GenerationCompletion complete,
                                    const std::stop_token& stop) const {
  if (stop.stop_requested()) {
    return std::unexpected(generation_cancelled_error());
  }
  if (!complete) {
    return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                 .message = "Async generation requires a completion handler."});
  }
  auto http = make_http_request(request, false);
  if (!http) {
    return std::unexpected(http.error());
  }
  return cail::detail::send_generation_async(*transport_, std::move(request), std::move(*http),
                                             std::move(complete), stop, decode_http_response);
}

Result<void> Client::stream_async(GenerationRequest request, StreamHandler on_event,
                                  LanguageModel::GenerationCompletion complete,
                                  const std::stop_token& stop) const {
  if (!on_event || !complete) {
    return std::unexpected(
        Error{.code = ErrorCode::invalid_configuration,
              .message = "Async streaming requires event and completion handlers."});
  }
  if (stop.stop_requested()) {
    return std::unexpected(generation_cancelled_error());
  }
  auto http = make_http_request(request, true);
  if (!http) {
    return std::unexpected(http.error());
  }
  auto state = std::make_shared<StreamState>();
  state->on_event = std::move(on_event);
  state->stop = stop;
  return cail::detail::stream_generation_async(
      *transport_, std::move(request), std::move(*http),
      [state](std::string_view bytes) { state->feed(bytes); }, std::move(complete), stop,
      [state](const HttpResponse& response) { return state->finish(response); });
}

Result<GenerationResponse> Client::decode_http_response(const HttpResponse& response) {
  const auto context = [&](Error error) {
    return unexpected_with_http_context<GenerationResponse>(std::move(error), response);
  };
  if (is_http_error_status(response.status_code)) {
    return context(http_status_error_from_json_body(response));
  }
  ResponseBody parsed;
  if (const auto error =
          glz::read<glz::opts{.error_on_unknown_keys = false}>(parsed, response.body);
      error) {
    return context(Error{.code = ErrorCode::provider_response,
                         .message = glz::format_error(error, response.body)});
  }
  auto result = decode(parsed);
  return result ? result : context(result.error());
}

Result<HttpRequest> Client::make_http_request(const GenerationRequest& request,
                                              bool streaming) const {
  if (config_.api_key.empty() || config_.model.empty() || config_.base_url.empty() ||
      (config_.max_tokens == 0 && !request.max_output_tokens) || !transport_) {
    return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                 .message = "Anthropic requires an API key, model, base URL, "
                                            "positive max_tokens, and transport."});
  }
  auto body = encode(request, config_, streaming);
  if (!body) {
    return std::unexpected(body.error());
  }
  auto encoded = to_json(*body);
  if (!encoded) {
    return std::unexpected(encoded.error());
  }
  encoded = apply_message_options(*encoded, request);
  if (!encoded) {
    return std::unexpected(encoded.error());
  }
  if (!request.provider_options.empty()) {
    auto merged = merge_json_objects(*encoded, request.provider_options);
    if (!merged) {
      return std::unexpected(merged.error());
    }
    encoded = std::move(*merged);
  }
  HttpRequest http{
      .url = config_.base_url + "/messages",
      .headers = config_.headers,
      .body = std::move(*encoded),
  };
  cail::detail::append_session_header(http.headers, config_.request_session_header,
                                      request.session_id);
  http.headers.push_back({.name = "x-api-key", .value = config_.api_key});
  http.headers.push_back({.name = "Authorization", .value = "Bearer " + config_.api_key});
  http.headers.push_back({.name = "anthropic-version", .value = "2023-06-01"});
  http.headers.push_back({.name = "Content-Type", .value = "application/json"});
  if (streaming) {
    http.headers.push_back({.name = "Accept", .value = "text/event-stream"});
  }
  if (auto middleware = cail::detail::run_before_request(request, http); !middleware) {
    return std::unexpected(middleware.error());
  }
  return http;
}

Result<GenerationResponse> Client::run(const GenerationRequest& request,
                                       const StreamHandler& on_event,
                                       const std::stop_token& stop) const {
  if (stop.stop_requested()) {
    return std::unexpected(generation_cancelled_error());
  }
  auto prepared = make_http_request(request, static_cast<bool>(on_event));
  if (!prepared) {
    return std::unexpected(prepared.error());
  }
  auto http = std::move(*prepared);
  StreamState state;
  state.on_event = on_event;
  state.stop = stop;
  auto response = on_event
                      ? transport_->stream(
                            http, [&state](std::string_view bytes) { state.feed(bytes); }, stop)
                      : transport_->send(http, stop);
  if (!response) {
    return std::unexpected(response.error());
  }
  if (auto middleware = cail::detail::run_after_response(request, *response); !middleware) {
    return std::unexpected(middleware.error());
  }
  if (stop.stop_requested()) {
    return std::unexpected(generation_cancelled_error());
  }
  return on_event ? state.finish(*response) : decode_http_response(*response);
}

} // namespace cail::detail::anthropic

namespace cail {

LanguageModel AnthropicProvider::operator()(std::string model_id) const {
  return (*this)(std::move(model_id), make_default_http_transport());
}

LanguageModel AnthropicProvider::operator()(std::string model_id,
                                            std::unique_ptr<HttpTransport> transport) const {
  auto key = detail::env_or(settings_.api_key, "ANTHROPIC_API_KEY");
  auto client = std::make_shared<detail::anthropic::Client>(
      detail::anthropic::Config{.api_key = std::move(key),
                                .model = std::move(model_id),
                                .base_url = settings_.base_url,
                                .max_tokens = settings_.max_tokens,
                                .headers = settings_.headers},
      std::move(transport));
  return LanguageModel{
      [client](const GenerationRequest& request, const std::stop_token& stop) {
        return client->generate(request, stop);
      },
      [client](const GenerationRequest& request, const StreamHandler& handler,
               const std::stop_token& stop) { return client->stream(request, handler, stop); },
      AdapterCapabilities{.image_input = true,
                          .pdf_input = true,
                          .tools = true,
                          .structured_output = true,
                          .reasoning = true},
      [client](GenerationRequest request, LanguageModel::GenerationCompletion complete,
               const std::stop_token& stop) {
        return client->generate_async(
            std::move(request),
            [client, complete = std::move(complete)](Result<GenerationResponse> result) {
              complete(std::move(result));
            },
            stop);
      },
      [client](GenerationRequest request, StreamHandler handler,
               LanguageModel::GenerationCompletion complete, const std::stop_token& stop) {
        return client->stream_async(
            std::move(request), std::move(handler),
            [client, complete = std::move(complete)](Result<GenerationResponse> result) {
              complete(std::move(result));
            },
            stop);
      },
  };
}

EmbeddingModel AnthropicProvider::embedding_model(std::string model_id,
                                                  std::optional<std::size_t> dimensions) const {
  return detail::make_openai_style_embedding_model(
      settings_.base_url + "/embeddings", detail::env_or(settings_.api_key, "ANTHROPIC_API_KEY"),
      std::move(model_id), dimensions);
}

} // namespace cail
