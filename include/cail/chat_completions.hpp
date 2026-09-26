#pragma once

#include <cail/detail/base64.hpp>
#include <cail/detail/glaze_http_transport.hpp>
#include <cail/detail/http_context.hpp>
#include <cail/detail/request_headers.hpp>
#include <cail/detail/sse.hpp>
#include <cail/detail/strict_schema.hpp>
#include <cail/generation.hpp>
#include <cail/json.hpp>
#include <cail/language_model.hpp>

#include <glaze/glaze.hpp>

#include <algorithm>
#include <map>
#include <memory>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cail::detail::chat_completions {

struct TextPart {
    std::string type{"text"};
    std::string text;
};
struct ImageUrl {
    std::string url;
};
struct ImagePart {
    std::string type{"image_url"};
    ImageUrl image_url;
};
struct FunctionCall {
    std::string name;
    std::string arguments;
};
struct ToolCall {
    std::optional<std::size_t> index;
    std::string id;
    std::string type{"function"};
    FunctionCall function;
};
struct ToolDefinition {
    std::string type{"function"};
    struct Function {
        std::string name;
        std::string description;
        glz::raw_json parameters;
    } function;
};
struct InputMessage {
    std::string role;
    std::optional<glz::raw_json> content;
    std::optional<std::string> tool_call_id;
    std::optional<std::vector<ToolCall>> tool_calls;
};
struct RequestBody {
    std::string model;
    std::vector<glz::raw_json> messages;
    std::optional<std::vector<glz::raw_json>> tools;
    std::optional<bool> stream;
    struct StreamOptions {
        bool include_usage{true};
    };
    std::optional<StreamOptions> stream_options;
    struct JsonSchema {
        std::string name;
        std::optional<std::string> description;
        bool strict{true};
        glz::raw_json schema;
    };
    struct ResponseFormat {
        std::string type{"json_schema"};
        JsonSchema json_schema;
    };
    std::optional<ResponseFormat> response_format;
    std::optional<std::size_t> max_tokens;
};
struct Usage {
    std::size_t prompt_tokens{};
    std::size_t completion_tokens{};
    struct Details {
        std::optional<std::size_t> cached_tokens;
        std::optional<std::size_t> reasoning_tokens;
    };
    std::optional<Details> prompt_tokens_details;
    std::optional<Details> completion_tokens_details;
};
struct ProviderError {
    std::string message;
    std::optional<std::string> code;
    std::optional<std::string> type;
};
struct ErrorBody {
    std::optional<ProviderError> error;
};
struct OutputToolCall {
    std::optional<std::size_t> index;
    std::optional<std::string> id;
    struct Function {
        std::optional<std::string> name;
        std::optional<std::string> arguments;
    };
    std::optional<Function> function;
};
struct OutputMessage {
    std::optional<glz::raw_json> content;
    std::optional<std::string> refusal;
    std::optional<std::string> reasoning_content;
    std::optional<glz::raw_json> reasoning_details;
    std::optional<std::vector<OutputToolCall>> tool_calls;
};
struct Choice {
    std::size_t index{};
    std::optional<std::string> finish_reason;
    std::optional<OutputMessage> message;
    std::optional<OutputMessage> delta;
};
struct ResponseBody {
    std::vector<Choice> choices;
    std::optional<Usage> usage;
    std::optional<ProviderError> error;
};

inline void append_reasoning_text(const glz::generic& value, std::string& text)
{
    if (const auto* string = value.get_if<std::string>()) {
        text += *string;
    } else if (const auto* array = value.get_if<glz::generic::array_t>()) {
        for (const auto& item : *array) append_reasoning_text(item, text);
    } else if (value.is_object()) {
        for (const auto* key : {"text", "content", "reasoning", "reasoning_content"}) {
            if (value.contains(key)) append_reasoning_text(value[key], text);
        }
    }
}

struct ContentText { std::string text; std::string reasoning; };

inline void append_content(const glz::generic& value, ContentText& output, bool thinking = false)
{
    if (const auto* string = value.get_if<std::string>()) {
        (thinking ? output.reasoning : output.text) += *string;
    } else if (const auto* array = value.get_if<glz::generic::array_t>()) {
        for (const auto& item : *array) append_content(item, output, thinking);
    } else if (value.is_object()) {
        if (value.contains("type")) {
            if (const auto* type = value["type"].get_if<std::string>())
                thinking |= *type == "thinking" || *type == "reasoning";
        }
        for (const auto* key : {"thinking", "text", "content"}) {
            if (value.contains(key)) append_content(value[key], output, thinking || std::string_view(key) == "thinking");
        }
    }
}

[[nodiscard]] inline Result<ContentText> decode_content(const glz::raw_json& raw)
{
    glz::generic value;
    if (const auto error = glz::read_json(value, raw.str); error)
        return std::unexpected(Error{.code = ErrorCode::provider_response,
                                     .message = glz::format_error(error, raw.str)});
    ContentText output;
    append_content(value, output);
    return output;
}

[[nodiscard]] inline TokenUsage usage(const Usage& value)
{
    return TokenUsage{
        .input_tokens = value.prompt_tokens,
        .output_tokens = value.completion_tokens,
        .cache_read_tokens = value.prompt_tokens_details ? value.prompt_tokens_details->cached_tokens : std::nullopt,
        .reasoning_tokens = value.completion_tokens_details ? value.completion_tokens_details->reasoning_tokens : std::nullopt,
    };
}

[[nodiscard]] inline std::optional<std::string> round_trip_options(
    bool retain_reasoning_content, std::string_view reasoning_content,
    const std::optional<std::string>& reasoning_details_json)
{
    glz::generic options = glz::generic::object_t{};
    if (retain_reasoning_content && !reasoning_content.empty()) {
        options["reasoning_content"] = std::string{reasoning_content};
    }
    if (reasoning_details_json && !reasoning_details_json->empty()) {
        glz::generic details;
        if (const auto error = glz::read_json(details, *reasoning_details_json); error) {
            return std::nullopt;
        }
        options["reasoning_details"] = std::move(details);
    }
    if (!options.is_object() || options.get<glz::generic::object_t>().empty()) {
        return std::nullopt;
    }
    auto encoded = options.dump();
    if (!encoded) {
        return std::nullopt;
    }
    return std::move(*encoded);
}

[[nodiscard]] inline std::optional<std::string> without_reasoning_content(
    const std::optional<std::string>& provider_options)
{
    if (!provider_options) {
        return std::nullopt;
    }
    glz::generic value;
    if (const auto error = glz::read_json(value, *provider_options); error || !value.is_object()) {
        return provider_options;
    }
    auto& object = value.get<glz::generic::object_t>();
    object.erase("reasoning_content");
    if (object.empty()) {
        return std::nullopt;
    }
    auto encoded = value.dump();
    if (!encoded) {
        return provider_options;
    }
    return std::move(*encoded);
}

[[nodiscard]] inline Result<GenerationResponse> decode(const ResponseBody& body, bool retain_reasoning_content)
{
    if (body.choices.size() != 1 || body.choices.front().index != 0 || !body.choices.front().message) {
        return std::unexpected(Error{.code = ErrorCode::provider_response,
                                     .message = "Chat Completions returned no single primary choice."});
    }
    const auto& choice = body.choices.front();
    const auto& message = *choice.message;
    GenerationResponse result;
    if (message.content) {
        auto content = decode_content(*message.content);
        if (!content) return std::unexpected(content.error());
        result.text = std::move(content->text);
        result.reasoning = std::move(content->reasoning);
    }
    if (message.refusal) result.text = *message.refusal;
    result.reasoning += message.reasoning_content.value_or("");
    result.provider_options = round_trip_options(
        retain_reasoning_content, message.reasoning_content.value_or(""),
        message.reasoning_details ? std::optional<std::string>{message.reasoning_details->str} : std::nullopt);
    if (message.refusal) {
        result.status = GenerationStatus::refused;
    } else if (choice.finish_reason == "length") {
        result.status = GenerationStatus::incomplete;
    } else if (choice.finish_reason != "stop" && choice.finish_reason != "tool_calls") {
        return std::unexpected(Error{.code = ErrorCode::provider_response,
                                     .message = "Chat Completions returned an unsupported finish reason."});
    }
    if (message.tool_calls) {
        for (const auto& call : *message.tool_calls) {
            if (!call.id || call.id->empty() || !call.function ||
                !call.function->name || call.function->name->empty()) {
                return std::unexpected(Error{.code = ErrorCode::provider_response,
                                             .message = "Chat Completions returned an incomplete tool call."});
            }
            result.tool_calls.push_back(cail::ToolCall{.id = *call.id, .name = *call.function->name,
                                                       .arguments = call.function->arguments.value_or("")});
        }
    }
    if (body.usage) {
        result.usage = usage(*body.usage);
    }
    return result;
}

[[nodiscard]] inline Result<RequestBody> encode(const GenerationRequest& request, std::string model,
    bool streaming, bool retain_reasoning_content)
{
    if (auto valid = cail::detail::validate_max_output_tokens(request); !valid) {
        return std::unexpected(valid.error());
    }
    if (request.messages.empty() || model.empty()) {
        return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                     .message = "Chat Completions requires a model and at least one message."});
    }
    if (request.continuation_token) {
        return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                     .message = "Chat Completions does not support continuation tokens."});
    }
    RequestBody body{.model = std::move(model)};
    body.max_tokens = request.max_output_tokens;
    if (streaming) {
        body.stream = true;
        if (request.stream_usage.value_or(true)) {
            body.stream_options = RequestBody::StreamOptions{};
        }
    }
    if (request.structured_output) {
        auto schema = cail::detail::strict_json_schema(request.structured_output->schema);
        if (!schema) return std::unexpected(schema.error());
        auto encoded = to_json(*schema);
        if (!encoded) return std::unexpected(encoded.error());
        body.response_format = RequestBody::ResponseFormat{
            .json_schema = RequestBody::JsonSchema{
                .name = request.structured_output->name,
                .description = request.structured_output->description,
                .schema = glz::raw_json{std::move(*encoded)},
            },
        };
    }
    const auto append_message = [&body, retain_reasoning_content](InputMessage item,
                                         std::optional<std::string> provider_options) -> Result<void> {
        if (!retain_reasoning_content) {
            provider_options = without_reasoning_content(provider_options);
        }
        auto encoded = to_json(item);
        if (!encoded) return std::unexpected(encoded.error());
        if (provider_options) {
            auto merged = merge_json_objects(*encoded, *provider_options);
            if (!merged) return std::unexpected(merged.error());
            encoded = std::move(*merged);
        }
        body.messages.emplace_back(std::move(*encoded));
        return {};
    };
    std::vector<glz::raw_json> pending_tool_images;
    const auto flush_tool_images = [&]() -> Result<void> {
        if (pending_tool_images.empty()) return {};
        auto encoded = to_json(pending_tool_images);
        if (!encoded) return std::unexpected(encoded.error());
        auto user_message = to_json(InputMessage{
            .role = "user", .content = glz::raw_json{std::move(*encoded)}});
        if (!user_message) return std::unexpected(user_message.error());
        body.messages.emplace_back(std::move(*user_message));
        pending_tool_images.clear();
        return {};
    };
    for (const auto& message : request.messages) {
        if (message.role != MessageRole::tool) {
            if (auto flushed = flush_tool_images(); !flushed)
                return std::unexpected(flushed.error());
        }
        InputMessage item;
        switch (message.role) {
        case MessageRole::system: item.role = "system"; break;
        case MessageRole::developer: item.role = "developer"; break;
        case MessageRole::user: item.role = "user"; break;
        case MessageRole::assistant: item.role = "assistant"; break;
        case MessageRole::tool: item.role = "tool"; break;
        }
        if (message.role == MessageRole::tool) {
            if (message.tool_call_id.empty() || !message.tool_calls.empty()) {
                return std::unexpected(Error{.code = ErrorCode::invalid_tool_call,
                                             .message = "A tool result requires a tool call ID and cannot contain tool calls."});
            }
            item.tool_call_id = message.tool_call_id;
        } else if (!message.tool_call_id.empty()) {
            return std::unexpected(Error{.code = ErrorCode::invalid_tool_call,
                                         .message = "Only tool results can set a tool call ID."});
        }
        if (!message.tool_calls.empty()) {
            if (message.role != MessageRole::assistant) {
                return std::unexpected(Error{.code = ErrorCode::invalid_tool_call,
                                             .message = "Only assistant messages can contain tool calls."});
            }
            item.tool_calls.emplace();
            for (const auto& call : message.tool_calls) {
                if (call.id.empty() || call.name.empty() || call.arguments.empty()) {
                    return std::unexpected(Error{.code = ErrorCode::invalid_tool_call,
                                                 .message = "Assistant tool calls require an ID, name, and arguments."});
                }
                item.tool_calls->push_back(ToolCall{.id = call.id,
                                                    .function = FunctionCall{.name = call.name,
                                                                             .arguments = call.arguments}});
            }
        }
        if (message.role == MessageRole::tool) {
            std::string text;
            for (const auto& part : message.content) {
                if (const auto* value = std::get_if<cail::TextPart>(&part)) {
                    text += value->text;
                } else {
                    const auto& image = std::get<cail::ImagePart>(part);
                    if (image.mime_type.empty() || image.bytes.empty()) {
                        return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                                     .message = "Tool result images require bytes and a MIME type."});
                    }
                    auto encoded = to_json(ImagePart{.image_url = ImageUrl{
                        .url = cail::detail::image_data_url(image.mime_type, image.bytes)}});
                    if (!encoded) return std::unexpected(encoded.error());
                    auto provider_part_options = image.provider_options;
                    if (provider_part_options) {
                        auto merged = merge_json_objects(*encoded, *provider_part_options);
                        if (!merged) return std::unexpected(merged.error());
                        encoded = std::move(*merged);
                    }
                    pending_tool_images.emplace_back(std::move(*encoded));
                }
            }
            auto encoded = to_json(text);
            if (!encoded) return std::unexpected(encoded.error());
            item.content = glz::raw_json{std::move(*encoded)};
        } else if (message.content.size() == 1 && std::holds_alternative<cail::TextPart>(message.content.front()) &&
                   !std::get<cail::TextPart>(message.content.front()).provider_options) {
            auto encoded = to_json(std::get<cail::TextPart>(message.content.front()).text);
            if (!encoded) return std::unexpected(encoded.error());
            item.content = glz::raw_json{std::move(*encoded)};
        } else if (!message.content.empty()) {
            const bool has_image = std::any_of(message.content.begin(), message.content.end(), [](const auto& part) {
                return std::holds_alternative<cail::ImagePart>(part);
            });
            if (message.role != MessageRole::user && has_image) {
                return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                             .message = "Images are supported only in user and tool-result messages."});
            }
            std::vector<glz::raw_json> parts;
            for (const auto& part : message.content) {
                Result<std::string> encoded = std::unexpected(Error{});
                if (const auto* text = std::get_if<cail::TextPart>(&part)) {
                    encoded = to_json(TextPart{.text = text->text});
                    if (encoded && text->provider_options) {
                        encoded = merge_json_objects(*encoded, *text->provider_options);
                    }
                } else if (const auto* image = std::get_if<cail::ImagePart>(&part)) {
                    if (image->mime_type.empty()) {
                        return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                                     .message = "An image requires a MIME type."});
                    }
                    encoded = to_json(ImagePart{.image_url = ImageUrl{
                        .url = cail::detail::image_data_url(image->mime_type, image->bytes)}});
                    if (encoded && image->provider_options) {
                        encoded = merge_json_objects(*encoded, *image->provider_options);
                    }
                }
                if (!encoded) return std::unexpected(encoded.error());
                parts.emplace_back(std::move(*encoded));
            }
            auto encoded = to_json(parts);
            if (!encoded) return std::unexpected(encoded.error());
            item.content = glz::raw_json{std::move(*encoded)};
        }
        if (auto added = append_message(std::move(item), message.provider_options); !added)
            return std::unexpected(added.error());
    }
    if (auto flushed = flush_tool_images(); !flushed)
        return std::unexpected(flushed.error());
    if (!request.tools.empty()) {
        body.tools.emplace();
        for (const auto& tool : request.tools) {
            if (tool.name.empty()) {
                return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                             .message = "Tool names cannot be empty."});
            }
            auto parameters = to_json(tool.parameters);
            if (!parameters) return std::unexpected(parameters.error());
            auto encoded = to_json(ToolDefinition{.function = ToolDefinition::Function{
                .name = tool.name, .description = tool.description, .parameters = glz::raw_json{std::move(*parameters)}}});
            if (!encoded) return std::unexpected(encoded.error());
            if (tool.provider_options) {
                auto merged = merge_json_objects(*encoded, *tool.provider_options);
                if (!merged) return std::unexpected(merged.error());
                encoded = std::move(*merged);
            }
            body.tools->emplace_back(std::move(*encoded));
        }
    }
    return body;
}

class Client {
public:
    Client(std::string endpoint, std::string model, std::string api_key, std::vector<HttpHeader> headers,
           std::unique_ptr<HttpTransport> transport = std::make_unique<cail::detail::GlazeHttpTransport>(),
           std::string request_session_header = {}, bool prompt_cache_key = false, bool session_body = false,
           bool retain_reasoning_content = true)
        : endpoint_(std::move(endpoint)), model_(std::move(model)), api_key_(std::move(api_key)),
          headers_(std::move(headers)), transport_(std::move(transport)),
          request_session_header_(std::move(request_session_header)), prompt_cache_key_(prompt_cache_key),
          session_body_(session_body), retain_reasoning_content_(retain_reasoning_content) {}

    [[nodiscard]] Result<GenerationResponse> generate(const GenerationRequest& request) const
    {
        return run(request, {}, {});
    }
    [[nodiscard]] Result<GenerationResponse> stream(const GenerationRequest& request,
        const StreamHandler& on_event, std::stop_token stop = {}) const
    {
        return run(request, on_event, stop);
    }

private:
    [[nodiscard]] Result<GenerationResponse> run(const GenerationRequest& request,
        const StreamHandler& on_event, std::stop_token stop) const
    {
        if (stop.stop_requested()) return std::unexpected(generation_cancelled_error());
        if (endpoint_.empty() || !transport_) return std::unexpected(Error{
            .code = ErrorCode::invalid_configuration, .message = "Chat Completions requires an endpoint and transport."});
        auto body = encode(request, model_, static_cast<bool>(on_event), retain_reasoning_content_);
        if (!body) return std::unexpected(body.error());
        auto encoded = to_json(*body);
        if (!encoded) return std::unexpected(encoded.error());
        if (!request.session_id.empty() && (prompt_cache_key_ || session_body_)) {
            glz::generic routing = glz::generic::object_t{};
            if (session_body_) routing["session_id"] = request.session_id;
            if (prompt_cache_key_) routing["prompt_cache_key"] = request.session_id;
            auto dumped = routing.dump();
            if (!dumped) return std::unexpected(Error{.code = ErrorCode::json_serialization,
                .message = "Could not encode Chat Completions session routing."});
            auto merged = merge_json_objects(*encoded, *dumped);
            if (!merged) return std::unexpected(merged.error());
            encoded = std::move(*merged);
        }
        if (request.provider_options) {
            auto merged = merge_json_objects(*encoded, *request.provider_options);
            if (!merged) return std::unexpected(merged.error());
            encoded = std::move(*merged);
        }
        HttpRequest http{.url = endpoint_, .headers = headers_, .body = std::move(*encoded)};
        http.headers.push_back({.name = "Content-Type", .value = "application/json"});
        if (!api_key_.empty()) http.headers.push_back({.name = "Authorization", .value = "Bearer " + api_key_});
        if (on_event) http.headers.push_back({.name = "Accept", .value = "text/event-stream"});
        cail::detail::append_session_header(http.headers, request_session_header_, request.session_id);
        if (request.before_request) {
            try {
                request.before_request(http);
            } catch (const std::exception& error) {
                return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                    .message = std::string{"The before-request callback failed: "} + error.what()});
            } catch (...) {
                return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                    .message = "The before-request callback failed."});
            }
        }
        cail::detail::SseParser parser;
        GenerationResponse partial;
        std::map<std::size_t, cail::ToolCall> pending_calls;
        std::vector<glz::raw_json> reasoning_details;
        std::string echoed_reasoning;
        bool finished = false;
        bool done = false;
        std::optional<Error> stream_error;
        const auto handle_event = [&](const cail::detail::ServerSentEvent& event) {
            if (stop.stop_requested() || stream_error || event.data.empty()) return;
            if (event.data == "[DONE]") { done = true; return; }
            ResponseBody chunk;
            if (const auto error = glz::read<glz::opts{.error_on_unknown_keys = false}>(chunk, event.data); error) {
                stream_error = Error{.code = ErrorCode::provider_response,
                                     .message = glz::format_error(error, event.data)};
                return;
            }
            if (chunk.error) {
                stream_error = Error{.code = ErrorCode::provider_response, .message = chunk.error->message,
                                     .provider_code = chunk.error->code.value_or(""),
                                     .provider_type = chunk.error->type.value_or("")};
                return;
            }
            if (chunk.usage) {
                partial.usage = usage(*chunk.usage);
                on_event(StreamEvent{UsageUpdate{.usage = *partial.usage}});
            }
            for (const auto& choice : chunk.choices) {
                if (choice.index != 0) continue;
                if (choice.delta) {
                    const auto& delta = *choice.delta;
                    if (delta.content) {
                        auto content = decode_content(*delta.content);
                        if (!content) { stream_error = content.error(); return; }
                        if (!content->reasoning.empty()) {
                            partial.reasoning += content->reasoning;
                            on_event(StreamEvent{ReasoningDelta{.text = content->reasoning}});
                        }
                        if (!content->text.empty()) {
                            partial.text += content->text;
                            on_event(StreamEvent{TextDelta{.text = content->text}});
                        }
                    }
                    if (delta.refusal) {
                        partial.status = GenerationStatus::refused;
                        partial.text += *delta.refusal;
                        on_event(StreamEvent{RefusalDelta{.text = *delta.refusal}});
                    }
                    if (delta.reasoning_content) {
                        partial.reasoning += *delta.reasoning_content;
                        echoed_reasoning += *delta.reasoning_content;
                        on_event(StreamEvent{ReasoningDelta{.text = *delta.reasoning_content}});
                    }
                    if (delta.reasoning_details) {
                        auto decoded = from_json<std::vector<glz::raw_json>>(delta.reasoning_details->str);
                        if (!decoded) {
                            stream_error = decoded.error();
                            return;
                        }
                        reasoning_details.insert(reasoning_details.end(), decoded->begin(), decoded->end());
                        std::string reasoning_text;
                        for (const auto& detail : *decoded) {
                            glz::generic value;
                            if (const auto error = glz::read_json(value, detail.str); error) {
                                stream_error = Error{.code = ErrorCode::provider_response,
                                    .message = glz::format_error(error, detail.str)};
                                return;
                            }
                            append_reasoning_text(value, reasoning_text);
                        }
                        if (!reasoning_text.empty()) {
                            on_event(StreamEvent{ReasoningDelta{.text = reasoning_text}});
                        }
                    }
                    if (delta.tool_calls) {
                        for (std::size_t index = 0; index < delta.tool_calls->size(); ++index) {
                            const auto& call = (*delta.tool_calls)[index];
                            const auto call_index = call.index.value_or(index);
                            auto& pending = pending_calls[call_index];
                            if (call.id && !call.id->empty()) pending.id = *call.id;
                            if (!call.function) continue;
                            if (call.function->name && !call.function->name->empty())
                                pending.name = *call.function->name;
                            if (call.function->arguments && !call.function->arguments->empty()) {
                                pending.arguments += *call.function->arguments;
                                on_event(StreamEvent{ToolCallArgumentsDelta{
                                    .output_index = call_index, .arguments = *call.function->arguments}});
                            }
                        }
                    }
                }
                if (choice.finish_reason) {
                    finished = true;
                    if (*choice.finish_reason == "length") partial.status = GenerationStatus::incomplete;
                    else if (*choice.finish_reason != "stop" && *choice.finish_reason != "tool_calls") {
                        stream_error = Error{.code = ErrorCode::provider_response,
                                             .message = "Chat Completions returned an unsupported finish reason."};
                    }
                }
            }
        };
        auto response = on_event ? transport_->stream(http, [&](std::string_view bytes) {
            parser.feed(bytes, handle_event);
        }, stop) : transport_->send(http);
        if (!response) return std::unexpected(response.error());
        if (request.after_response) {
            try {
                request.after_response(*response);
            } catch (const std::exception& error) {
                return std::unexpected(Error{.code = ErrorCode::provider_response,
                    .message = std::string{"The after-response callback failed: "} + error.what()});
            } catch (...) {
                return std::unexpected(Error{.code = ErrorCode::provider_response,
                    .message = "The after-response callback failed."});
            }
        }
        if (stop.stop_requested()) return std::unexpected(generation_cancelled_error());
        const auto context = [&](Error error) {
            return unexpected_with_http_context<GenerationResponse>(std::move(error), *response);
        };
        if (response->status_code < 200 || response->status_code >= 300) {
            ErrorBody error_body;
            const auto parsed = glz::read<glz::opts{.error_on_unknown_keys = false}>(error_body, response->body);
            return context(Error{.code = ErrorCode::http_status,
                .message = !parsed && error_body.error ? error_body.error->message : response->body,
                .provider_code = !parsed && error_body.error ? error_body.error->code.value_or("") : "",
                .provider_type = !parsed && error_body.error ? error_body.error->type.value_or("") : ""});
        }
        if (!on_event) {
            ResponseBody parsed;
            if (const auto error = glz::read<glz::opts{.error_on_unknown_keys = false}>(parsed, response->body); error)
                return context(Error{.code = ErrorCode::provider_response,
                                     .message = glz::format_error(error, response->body)});
            auto result = decode(parsed, retain_reasoning_content_);
            return result ? result : context(result.error());
        }
        parser.finish(handle_event);
        if (stream_error) return context(*stream_error);
        if (!done || !finished) return context(Error{.code = ErrorCode::provider_response,
            .message = "Chat Completions closed before the final choice and [DONE]."});
        for (const auto& [index, call] : pending_calls) {
            if (call.id.empty() || call.name.empty()) return context(Error{
                .code = ErrorCode::provider_response, .message = "Chat Completions returned an incomplete tool call."});
            partial.tool_calls.push_back(call);
            on_event(StreamEvent{ToolCallReady{.output_index = index, .call = call}});
        }
        std::optional<std::string> details_json;
        if (!reasoning_details.empty()) {
            auto encoded_details = to_json(reasoning_details);
            if (!encoded_details) return context(encoded_details.error());
            details_json = std::move(*encoded_details);
        }
        partial.provider_options = round_trip_options(retain_reasoning_content_, echoed_reasoning, details_json);
        return partial;
    }

    std::string endpoint_;
    std::string model_;
    std::string api_key_;
    std::vector<HttpHeader> headers_;
    std::unique_ptr<HttpTransport> transport_;
    std::string request_session_header_;
    bool prompt_cache_key_ = false;
    bool session_body_ = false;
    bool retain_reasoning_content_ = true;
};

} // namespace cail::detail::chat_completions

namespace cail {

struct ChatCompletionsSettings {
    std::string endpoint{"https://api.openai.com/v1/chat/completions"};
    std::string api_key;
    std::vector<HttpHeader> headers;
    std::string request_session_header;
    bool prompt_cache_key = false;
    bool session_body = false;
    bool retain_reasoning_content = true;
};

[[nodiscard]] constexpr AdapterCapabilities chat_completions_adapter_capabilities()
{
    return AdapterCapabilities{
        .image_input = true,
        .tools = true,
        .structured_output = true,
        .reasoning = true,
    };
}

class ChatCompletionsProvider {
public:
    explicit ChatCompletionsProvider(ChatCompletionsSettings settings) : settings_(std::move(settings)) {}
    [[nodiscard]] LanguageModel operator()(std::string model_id) const
    {
        return (*this)(std::move(model_id), std::make_unique<detail::GlazeHttpTransport>());
    }

    [[nodiscard]] LanguageModel operator()(std::string model_id,
                                           std::unique_ptr<HttpTransport> transport) const
    {
        auto client = std::make_shared<detail::chat_completions::Client>(
            settings_.endpoint, std::move(model_id), settings_.api_key, settings_.headers,
            std::move(transport), settings_.request_session_header, settings_.prompt_cache_key,
            settings_.session_body, settings_.retain_reasoning_content);
        return LanguageModel{
            [client](const GenerationRequest& request) { return client->generate(request); },
            [client](const GenerationRequest& request, const StreamHandler& handler, std::stop_token stop) {
                return client->stream(request, handler, stop);
            }, chat_completions_adapter_capabilities()};
    }
private:
    ChatCompletionsSettings settings_;
};

[[nodiscard]] inline ChatCompletionsProvider create_chat_completions(ChatCompletionsSettings settings)
{
    return ChatCompletionsProvider{std::move(settings)};
}

} // namespace cail
