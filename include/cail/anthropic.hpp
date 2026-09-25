#pragma once

#include <cail/detail/base64.hpp>
#include <cail/detail/encode_json.hpp>
#include <cail/detail/env.hpp>
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

namespace cail::detail::anthropic {

struct Config {
    std::string api_key;
    std::string model;
    std::string base_url{"https://api.anthropic.com/v1"};
    std::size_t max_tokens{1024};
    std::vector<HttpHeader> headers;
};

struct TextBlock {
    std::string type{"text"};
    std::string text;
};
struct ImageSource {
    std::string type{"base64"};
    std::string media_type;
    std::string data;
};
struct ImageBlock {
    std::string type{"image"};
    ImageSource source;
};
struct ToolUseBlock {
    std::string type{"tool_use"};
    std::string id;
    std::string name;
    glz::raw_json input;
};
struct ToolResultBlock {
    std::string type{"tool_result"};
    std::string tool_use_id;
    std::string content;
};
struct InputMessage {
    std::string role;
    std::vector<glz::raw_json> content;
};
struct Tool {
    std::string name;
    std::string description;
    glz::raw_json input_schema;
};
struct RequestBody {
    std::string model;
    std::size_t max_tokens{};
    std::vector<InputMessage> messages;
    std::optional<std::string> system;
    std::optional<std::vector<Tool>> tools;
    std::optional<bool> stream;
    struct OutputConfig {
        struct Format {
            std::string type{"json_schema"};
            glz::raw_json schema;
        } format;
    };
    std::optional<OutputConfig> output_config;
};
struct Usage {
    std::optional<std::size_t> input_tokens;
    std::optional<std::size_t> output_tokens;
    std::optional<std::size_t> cache_read_input_tokens;
    std::optional<std::size_t> cache_creation_input_tokens;
};
struct OutputBlock {
    std::string type;
    std::optional<std::string> text;
    std::optional<std::string> thinking;
    std::optional<std::string> id;
    std::optional<std::string> name;
    glz::raw_json input;
};
struct ResponseBody {
    std::vector<OutputBlock> content;
    std::optional<std::string> stop_reason;
    std::optional<Usage> usage;
};
struct ProviderError {
    std::string type;
    std::string message;
};
struct ErrorBody {
    std::optional<ProviderError> error;
};
struct StreamDelta {
    std::string type;
    std::optional<std::string> text;
    std::optional<std::string> thinking;
    std::optional<std::string> partial_json;
    std::optional<std::string> stop_reason;
};
struct StreamBody {
    std::string type;
    std::optional<std::size_t> index;
    std::optional<OutputBlock> content_block;
    std::optional<StreamDelta> delta;
    std::optional<ResponseBody> message;
    std::optional<Usage> usage;
    std::optional<ProviderError> error;
};

[[nodiscard]] inline Result<RequestBody> encode(const GenerationRequest& request, const Config& config, bool streaming)
{
    if (request.messages.empty()) {
        return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                     .message = "Anthropic requires at least one message."});
    }
    if (request.continuation_token) {
        return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                     .message = "Anthropic does not support continuation tokens."});
    }
    RequestBody body{.model = config.model, .max_tokens = config.max_tokens};
    if (streaming) body.stream = true;
    if (request.structured_output) {
        auto schema = cail::detail::strict_schema(request.structured_output->schema);
        if (!schema) return std::unexpected(schema.error());
        auto encoded = to_json(*schema);
        if (!encoded) return std::unexpected(encoded.error());
        body.output_config = RequestBody::OutputConfig{
            .format = RequestBody::OutputConfig::Format{
                .schema = glz::raw_json{std::move(*encoded)},
            },
        };
    }
    for (const auto& message : request.messages) {
        if (message.role == MessageRole::system) {
            if (!body.messages.empty() || !message.tool_calls.empty() || !message.tool_call_id.empty()) {
                return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                             .message = "Anthropic system messages must precede conversation messages."});
            }
            for (const auto& part : message.content) {
                const auto* text = std::get_if<cail::TextPart>(&part);
                if (!text) return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                                         .message = "Anthropic system content must be text."});
                if (body.system) body.system->append("\n");
                else body.system.emplace();
                body.system->append(text->text);
            }
            continue;
        }
        if (message.role == MessageRole::developer) {
            return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                         .message = "Anthropic does not accept developer-role messages."});
        }
        InputMessage item{.role = message.role == MessageRole::assistant ? "assistant" : "user"};
        if (message.role == MessageRole::tool) {
            if (message.tool_call_id.empty() || !message.tool_calls.empty() || message.content.size() != 1 ||
                !std::holds_alternative<cail::TextPart>(message.content.front())) {
                return std::unexpected(Error{.code = ErrorCode::invalid_tool_call,
                                             .message = "Anthropic tool results require a call ID and one text part."});
            }
            auto added = append_json(item.content, ToolResultBlock{
                .tool_use_id = message.tool_call_id,
                .content = std::get<cail::TextPart>(message.content.front()).text});
            if (!added) return std::unexpected(added.error());
        } else {
            if (!message.tool_call_id.empty() ||
                (message.role != MessageRole::assistant && !message.tool_calls.empty())) {
                return std::unexpected(Error{.code = ErrorCode::invalid_tool_call,
                                             .message = "Anthropic found a misplaced tool call ID or tool call."});
            }
            for (const auto& part : message.content) {
                Result<void> added;
                if (const auto* text = std::get_if<cail::TextPart>(&part)) {
                    added = append_json(item.content, TextBlock{.text = text->text});
                } else {
                    const auto& image = std::get<cail::ImagePart>(part);
                    if (message.role != MessageRole::user || image.mime_type.empty() || image.bytes.empty()) {
                        return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                                     .message = "Anthropic images require user content, bytes, and a MIME type."});
                    }
                    added = append_json(item.content, ImageBlock{.source = ImageSource{
                        .media_type = image.mime_type, .data = cail::detail::base64_encode(image.bytes)}});
                }
                if (!added) return std::unexpected(added.error());
            }
            for (const auto& call : message.tool_calls) {
                if (call.id.empty() || call.name.empty() || call.arguments.empty() ||
                    glz::validate_json(call.arguments)) {
                    return std::unexpected(Error{.code = ErrorCode::invalid_tool_call,
                                                 .message = "Anthropic assistant tool calls require an ID, name, and JSON arguments."});
                }
                auto added = append_json(item.content, ToolUseBlock{
                    .id = call.id, .name = call.name, .input = glz::raw_json{call.arguments}});
                if (!added) return std::unexpected(added.error());
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
            if (!schema) return std::unexpected(schema.error());
            body.tools->push_back(Tool{.name = tool.name, .description = tool.description,
                                       .input_schema = glz::raw_json{std::move(*schema)}});
        }
    }
    return body;
}

inline void apply_usage(TokenUsage& target, const Usage& usage)
{
    if (usage.input_tokens) target.input_tokens = *usage.input_tokens;
    if (usage.output_tokens) target.output_tokens = *usage.output_tokens;
    if (usage.cache_read_input_tokens) target.cache_read_tokens = usage.cache_read_input_tokens;
    if (usage.cache_creation_input_tokens) target.cache_write_tokens = usage.cache_creation_input_tokens;
}

[[nodiscard]] inline Result<void> apply_stop_reason(GenerationResponse& result, const std::optional<std::string>& reason)
{
    if (reason == "max_tokens") result.status = GenerationStatus::incomplete;
    else if (reason == "refusal") result.status = GenerationStatus::refused;
    else if (reason != "end_turn" && reason != "tool_use" && reason != "stop_sequence") {
        return std::unexpected(Error{.code = ErrorCode::provider_response,
                                     .message = "Anthropic returned an unsupported stop reason."});
    }
    return {};
}

[[nodiscard]] inline Result<GenerationResponse> decode(const ResponseBody& body)
{
    GenerationResponse result;
    auto status = apply_stop_reason(result, body.stop_reason);
    if (!status) return std::unexpected(status.error());
    for (const auto& block : body.content) {
        if (block.type == "text" && block.text) result.text += *block.text;
        else if (block.type == "thinking" && block.thinking) result.reasoning += *block.thinking;
        else if (block.type == "tool_use") {
            if (!block.id || !block.name || block.input.str.empty()) {
                return std::unexpected(Error{.code = ErrorCode::provider_response,
                                             .message = "Anthropic returned an incomplete tool call."});
            }
            result.tool_calls.push_back(cail::ToolCall{
                .id = *block.id, .name = *block.name, .arguments = block.input.str});
        }
    }
    if (body.usage) {
        result.usage.emplace();
        apply_usage(*result.usage, *body.usage);
    }
    return result;
}

class Client {
    public:
    explicit Client(Config config, std::unique_ptr<HttpTransport> transport =
                                      std::make_unique<cail::detail::GlazeHttpTransport>())
        : config_(std::move(config)), transport_(std::move(transport)) {}

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
        if (config_.api_key.empty() || config_.model.empty() || config_.base_url.empty() ||
            config_.max_tokens == 0 || !transport_) {
            return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                         .message = "Anthropic requires an API key, model, base URL, positive max_tokens, and transport."});
        }
        auto body = encode(request, config_, static_cast<bool>(on_event));
        if (!body) return std::unexpected(body.error());
        auto encoded = to_json(*body);
        if (!encoded) return std::unexpected(encoded.error());
        HttpRequest http{
            .url = config_.base_url + "/messages",
            .headers = config_.headers,
            .body = std::move(*encoded),
        };
        http.headers.push_back({.name = "Authorization", .value = "Bearer " + config_.api_key});
        http.headers.push_back({.name = "anthropic-version", .value = "2023-06-01"});
        http.headers.push_back({.name = "Content-Type", .value = "application/json"});
        if (on_event) http.headers.push_back({.name = "Accept", .value = "text/event-stream"});
        cail::detail::SseParser parser;
        GenerationResponse partial;
        std::map<std::size_t, cail::ToolCall> pending_calls;
        bool finished = false;
        std::optional<Error> stream_error;
        const auto handle_event = [&](const cail::detail::ServerSentEvent& event) {
            if (stop.stop_requested() || stream_error || event.data.empty()) return;
            StreamBody chunk;
            if (const auto error = glz::read<glz::opts{.error_on_unknown_keys = false}>(chunk, event.data); error) {
                stream_error = Error{.code = ErrorCode::provider_response,
                                     .message = glz::format_error(error, event.data)};
                return;
            }
            if (chunk.type == "error") {
                stream_error = Error{.code = ErrorCode::provider_response,
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
            } else if (chunk.type == "content_block_delta" && chunk.index && chunk.delta) {
                const auto& delta = *chunk.delta;
                if (delta.type == "text_delta" && delta.text) {
                    partial.text += *delta.text;
                    on_event(StreamEvent{TextDelta{.text = *delta.text}});
                } else if (delta.type == "thinking_delta" && delta.thinking) {
                    partial.reasoning += *delta.thinking;
                    on_event(StreamEvent{ReasoningDelta{.text = *delta.thinking}});
                } else if (delta.type == "input_json_delta" && delta.partial_json) {
                    pending_calls[*chunk.index].arguments += *delta.partial_json;
                    on_event(StreamEvent{ToolCallArgumentsDelta{
                        .output_index = *chunk.index, .arguments = *delta.partial_json}});
                }
            } else if (chunk.type == "content_block_stop" && chunk.index) {
                if (auto found = pending_calls.find(*chunk.index); found != pending_calls.end()) {
                    if (found->second.arguments.empty()) found->second.arguments = "{}";
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
                    auto status = apply_stop_reason(partial, chunk.delta->stop_reason);
                    if (!status) stream_error = status.error();
                }
                if (chunk.usage) {
                    if (!partial.usage) partial.usage.emplace();
                    apply_usage(*partial.usage, *chunk.usage);
                    on_event(StreamEvent{UsageUpdate{.usage = *partial.usage}});
                }
            } else if (chunk.type == "message_stop") {
                finished = true;
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
                .provider_type = !parsed && error_body.error ? error_body.error->type : ""});
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
        if (!finished || !pending_calls.empty()) return context(Error{.code = ErrorCode::provider_response,
            .message = "Anthropic closed the stream before message_stop or a completed tool call."});
        return partial;
    }

    Config config_;
    std::unique_ptr<HttpTransport> transport_;
};

} // namespace cail::detail::anthropic

namespace cail {

struct AnthropicSettings {
    std::string api_key;
    std::string base_url{"https://api.anthropic.com/v1"};
    std::size_t max_tokens{1024};
    std::vector<HttpHeader> headers;
};

class AnthropicProvider {
    public:
    explicit AnthropicProvider(AnthropicSettings settings = {}) : settings_(std::move(settings)) {}

    [[nodiscard]] LanguageModel operator()(std::string model_id) const
    {
        auto key = detail::env_or(settings_.api_key, "ANTHROPIC_API_KEY");
        auto client = std::make_shared<detail::anthropic::Client>(detail::anthropic::Config{
            .api_key = std::move(key), .model = std::move(model_id), .base_url = settings_.base_url,
            .max_tokens = settings_.max_tokens, .headers = settings_.headers});
        return LanguageModel{
            [client](const GenerationRequest& request) { return client->generate(request); },
            [client](const GenerationRequest& request, const StreamHandler& handler, std::stop_token stop) {
                return client->stream(request, handler, stop);
            }, LanguageModelCapabilities{
                .image_input = true, .tools = true, .structured_output = true, .reasoning = true}};
    }

    private:
    AnthropicSettings settings_;
};

[[nodiscard]] inline AnthropicProvider create_anthropic(AnthropicSettings settings = {})
{
    return AnthropicProvider{std::move(settings)};
}

inline AnthropicProvider anthropic{};

} // namespace cail
