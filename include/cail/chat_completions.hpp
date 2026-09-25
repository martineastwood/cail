#pragma once

#include <cail/detail/base64.hpp>
#include <cail/detail/glaze_http_transport.hpp>
#include <cail/detail/http_context.hpp>
#include <cail/detail/sse.hpp>
#include <cail/detail/strict_schema.hpp>
#include <cail/generation.hpp>
#include <cail/json.hpp>
#include <cail/language_model.hpp>

#include <glaze/glaze.hpp>

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
    std::vector<InputMessage> messages;
    std::optional<std::vector<ToolDefinition>> tools;
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
struct OutputMessage {
    std::optional<std::string> content;
    std::optional<std::string> refusal;
    std::optional<std::string> reasoning_content;
    std::optional<std::vector<ToolCall>> tool_calls;
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

[[nodiscard]] inline TokenUsage usage(const Usage& value)
{
    return TokenUsage{
        .input_tokens = value.prompt_tokens,
        .output_tokens = value.completion_tokens,
        .cache_read_tokens = value.prompt_tokens_details ? value.prompt_tokens_details->cached_tokens : std::nullopt,
        .reasoning_tokens = value.completion_tokens_details ? value.completion_tokens_details->reasoning_tokens : std::nullopt,
    };
}

[[nodiscard]] inline Result<GenerationResponse> decode(const ResponseBody& body)
{
    if (body.choices.size() != 1 || body.choices.front().index != 0 || !body.choices.front().message) {
        return std::unexpected(Error{.code = ErrorCode::provider_response,
                                     .message = "Chat Completions returned no single primary choice."});
    }
    const auto& choice = body.choices.front();
    const auto& message = *choice.message;
    GenerationResponse result;
    result.text = message.refusal.value_or(message.content.value_or(""));
    result.reasoning = message.reasoning_content.value_or("");
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
            if (call.id.empty() || call.function.name.empty()) {
                return std::unexpected(Error{.code = ErrorCode::provider_response,
                                             .message = "Chat Completions returned an incomplete tool call."});
            }
            result.tool_calls.push_back(cail::ToolCall{.id = call.id, .name = call.function.name,
                                                       .arguments = call.function.arguments});
        }
    }
    if (body.usage) {
        result.usage = usage(*body.usage);
    }
    return result;
}

[[nodiscard]] inline Result<RequestBody> encode(const GenerationRequest& request, std::string model, bool streaming)
{
    if (request.messages.empty() || model.empty()) {
        return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                     .message = "Chat Completions requires a model and at least one message."});
    }
    if (request.continuation_token) {
        return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                     .message = "Chat Completions does not support continuation tokens."});
    }
    RequestBody body{.model = std::move(model)};
    if (streaming) {
        body.stream = true;
        body.stream_options = RequestBody::StreamOptions{};
    }
    if (request.structured_output) {
        auto schema = cail::detail::strict_schema(request.structured_output->schema);
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
    for (const auto& message : request.messages) {
        InputMessage item;
        switch (message.role) {
        case MessageRole::system: item.role = "system"; break;
        case MessageRole::developer: item.role = "developer"; break;
        case MessageRole::user: item.role = "user"; break;
        case MessageRole::assistant: item.role = "assistant"; break;
        case MessageRole::tool: item.role = "tool"; break;
        }
        if (message.role == MessageRole::tool) {
            if (message.tool_call_id.empty()) {
                return std::unexpected(Error{.code = ErrorCode::invalid_tool_call,
                                             .message = "A tool result requires a tool call ID."});
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
        if (message.role == MessageRole::tool && message.content.size() != 1) {
            return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                         .message = "A tool result requires one text content part."});
        }
        if (message.content.size() == 1 && std::holds_alternative<cail::TextPart>(message.content.front())) {
            auto encoded = to_json(std::get<cail::TextPart>(message.content.front()).text);
            if (!encoded) return std::unexpected(encoded.error());
            item.content = glz::raw_json{std::move(*encoded)};
        } else if (!message.content.empty()) {
            if (message.role != MessageRole::user) {
                return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                             .message = "Image and multipart content is supported only in user messages."});
            }
            std::vector<glz::raw_json> parts;
            for (const auto& part : message.content) {
                Result<std::string> encoded = std::unexpected(Error{});
                if (const auto* text = std::get_if<cail::TextPart>(&part)) {
                    encoded = to_json(TextPart{.text = text->text});
                } else if (const auto* image = std::get_if<cail::ImagePart>(&part)) {
                    if (image->mime_type.empty()) {
                        return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                                     .message = "An image requires a MIME type."});
                    }
                    encoded = to_json(ImagePart{.image_url = ImageUrl{
                        .url = cail::detail::image_data_url(image->mime_type, image->bytes)}});
                }
                if (!encoded) return std::unexpected(encoded.error());
                parts.emplace_back(std::move(*encoded));
            }
            auto encoded = to_json(parts);
            if (!encoded) return std::unexpected(encoded.error());
            item.content = glz::raw_json{std::move(*encoded)};
        }
        body.messages.push_back(std::move(item));
    }
    if (!request.tools.empty()) {
        body.tools.emplace();
        for (const auto& tool : request.tools) {
            if (tool.name.empty()) {
                return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                             .message = "Tool names cannot be empty."});
            }
            auto parameters = to_json(tool.parameters);
            if (!parameters) return std::unexpected(parameters.error());
            body.tools->push_back(ToolDefinition{.function = ToolDefinition::Function{
                .name = tool.name, .description = tool.description, .parameters = glz::raw_json{std::move(*parameters)}}});
        }
    }
    return body;
}

class Client {
public:
    Client(std::string endpoint, std::string model, std::string api_key, std::vector<HttpHeader> headers,
           std::unique_ptr<HttpTransport> transport = std::make_unique<cail::detail::GlazeHttpTransport>())
        : endpoint_(std::move(endpoint)), model_(std::move(model)), api_key_(std::move(api_key)),
          headers_(std::move(headers)), transport_(std::move(transport)) {}

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
        auto body = encode(request, model_, static_cast<bool>(on_event));
        if (!body) return std::unexpected(body.error());
        auto encoded = to_json(*body);
        if (!encoded) return std::unexpected(encoded.error());
        HttpRequest http{.url = endpoint_, .headers = headers_, .body = std::move(*encoded)};
        http.headers.push_back({.name = "Content-Type", .value = "application/json"});
        if (!api_key_.empty()) http.headers.push_back({.name = "Authorization", .value = "Bearer " + api_key_});
        if (on_event) http.headers.push_back({.name = "Accept", .value = "text/event-stream"});
        cail::detail::SseParser parser;
        GenerationResponse partial;
        std::map<std::size_t, cail::ToolCall> pending_calls;
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
                        partial.text += *delta.content;
                        on_event(StreamEvent{TextDelta{.text = *delta.content}});
                    }
                    if (delta.refusal) {
                        partial.status = GenerationStatus::refused;
                        partial.text += *delta.refusal;
                        on_event(StreamEvent{RefusalDelta{.text = *delta.refusal}});
                    }
                    if (delta.reasoning_content) {
                        partial.reasoning += *delta.reasoning_content;
                        on_event(StreamEvent{ReasoningDelta{.text = *delta.reasoning_content}});
                    }
                    if (delta.tool_calls) {
                        for (std::size_t index = 0; index < delta.tool_calls->size(); ++index) {
                            const auto& call = (*delta.tool_calls)[index];
                            const auto call_index = call.index.value_or(index);
                            auto& pending = pending_calls[call_index];
                            if (!call.id.empty()) pending.id = call.id;
                            if (!call.function.name.empty()) pending.name = call.function.name;
                            pending.arguments += call.function.arguments;
                            if (!call.function.arguments.empty()) on_event(StreamEvent{ToolCallArgumentsDelta{
                                .output_index = call_index, .arguments = call.function.arguments}});
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
            auto result = decode(parsed);
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
        return partial;
    }

    std::string endpoint_;
    std::string model_;
    std::string api_key_;
    std::vector<HttpHeader> headers_;
    std::unique_ptr<HttpTransport> transport_;
};

} // namespace cail::detail::chat_completions

namespace cail {

struct ChatCompletionsSettings {
    std::string endpoint{"https://api.openai.com/v1/chat/completions"};
    std::string api_key;
    std::vector<HttpHeader> headers;
};

class ChatCompletionsProvider {
public:
    explicit ChatCompletionsProvider(ChatCompletionsSettings settings) : settings_(std::move(settings)) {}
    [[nodiscard]] LanguageModel operator()(std::string model_id) const
    {
        auto client = std::make_shared<detail::chat_completions::Client>(
            settings_.endpoint, std::move(model_id), settings_.api_key, settings_.headers);
        return LanguageModel{
            [client](const GenerationRequest& request) { return client->generate(request); },
            [client](const GenerationRequest& request, const StreamHandler& handler, std::stop_token stop) {
                return client->stream(request, handler, stop);
            }, LanguageModelCapabilities{
                .image_input = true,
                .tools = true,
                .structured_output = true,
                .reasoning = true,
            }};
    }
private:
    ChatCompletionsSettings settings_;
};

[[nodiscard]] inline ChatCompletionsProvider create_chat_completions(ChatCompletionsSettings settings)
{
    return ChatCompletionsProvider{std::move(settings)};
}

} // namespace cail
