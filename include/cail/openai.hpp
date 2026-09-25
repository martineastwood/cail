#pragma once

#include <cail/detail/base64.hpp>
#include <cail/detail/glaze_http_transport.hpp>
#include <cail/detail/sse.hpp>
#include <cail/detail/strict_schema.hpp>
#include <cail/generation.hpp>
#include <cail/language_model.hpp>
#include <cail/openai_embeddings.hpp>
#include <cail/json.hpp>
#include <cail/tool.hpp>

#include <glaze/glaze.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <map>
#include <memory>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cail::detail::openai {

struct Config {
    std::string api_key;
    std::string model;
    std::string base_url{"https://api.openai.com/v1"};
};

namespace detail {

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
    std::vector<ResponseItem> output;
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

[[nodiscard]] inline Result<GenerationResponse> decode_response(ResponseBody response_body, int http_status)
{
    if (response_body.status == "failed") {
        return std::unexpected(Error{
            .code = ErrorCode::provider_response,
            .message = response_body.error ? response_body.error->message : "OpenAI reported a failed response.",
            .http_status = http_status,
            .provider_code = response_body.error && response_body.error->code ? *response_body.error->code : "",
            .provider_type = response_body.error && response_body.error->type ? *response_body.error->type : "",
        });
    }

    GenerationResponse result;
    result.continuation_token = response_body.id;
    if (response_body.status == "incomplete") {
        result.status = GenerationStatus::incomplete;
    } else if (response_body.status != "completed") {
        return std::unexpected(Error{
            .code = ErrorCode::provider_response,
            .message = "OpenAI returned an unsupported response status: " + response_body.status,
            .http_status = http_status,
        });
    }

    for (const auto& item : response_body.output) {
        if (item.type == "reasoning") {
            for (const auto& part : item.summary) {
                if (part.text) {
                    result.reasoning += *part.text;
                }
            }
            continue;
        }
        if (item.type == "function_call") {
            if (!item.call_id || item.call_id->empty() || !item.name || item.name->empty() || !item.arguments) {
                return std::unexpected(Error{
                    .code = ErrorCode::provider_response,
                    .message = "OpenAI returned a function call without its call ID, name, or arguments.",
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

[[nodiscard]] inline Result<std::string> text_content(const Message& message) {
    std::string text;
    for (const auto& part : message.content) {
        if (const auto* value = std::get_if<TextPart>(&part)) {
            text += value->text;
        } else {
            return std::unexpected(Error{
                .code = ErrorCode::invalid_configuration,
                .message = "OpenAI Responses supports image parts only in user messages.",
            });
        }
    }
    return text;
}

[[nodiscard]] inline Result<glz::raw_json> input_content(const Message& message) {
    const auto has_image = std::ranges::any_of(message.content, [](const ContentPart& part) {
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
            .message = "OpenAI Responses supports image parts only in user messages.",
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
        } else {
            const auto& image = std::get<ImagePart>(part);
            if (image.mime_type.empty() || image.bytes.empty()) {
                return std::unexpected(Error{
                    .code = ErrorCode::invalid_configuration,
                    .message = "An image part requires non-empty bytes and a MIME type.",
                });
            }
            encoded = to_json(InputImagePart{
                .image_url = "data:" + image.mime_type + ";base64," + cail::detail::base64_encode(image.bytes),
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


} // namespace detail

class Client {
    public:
    explicit Client(Config config) : Client(std::move(config), std::make_unique<cail::detail::GlazeHttpTransport>()) {}

    Client(Config config, std::unique_ptr<HttpTransport> transport)
        : config_(std::move(config)), transport_(std::move(transport)) {}

    [[nodiscard]] Result<GenerationResponse> generate(const GenerationRequest& request) const {
        return generate_impl(request, {});
    }

    [[nodiscard]] Result<GenerationResponse> stream(
        const GenerationRequest& request, const StreamHandler& on_event, std::stop_token stop = {}) const {
        if (!on_event) {
            return std::unexpected(Error{
                .code = ErrorCode::invalid_configuration,
                .message = "Streaming requires an event handler.",
            });
        }
        return generate_impl(request, on_event, stop);
    }

    [[nodiscard]] Result<GenerationResponse> stream(
        std::string_view prompt, const StreamHandler& on_event, std::stop_token stop = {}) const {
        return stream(GenerationRequest{
            .messages = {Message{.role = MessageRole::user, .content = {TextPart{.text = std::string{prompt}}}}},
        }, on_event, stop);
    }

    private:
    [[nodiscard]] Result<GenerationResponse> generate_impl(
        const GenerationRequest& request, const StreamHandler& on_event, std::stop_token stop = {}) const {
        const bool streaming = static_cast<bool>(on_event);
        if (stop.stop_requested()) {
            return std::unexpected(Error{.code = ErrorCode::cancelled, .message = "Generation was cancelled."});
        }
        if (config_.api_key.empty() || config_.model.empty() || config_.base_url.empty() || !transport_) {
            return std::unexpected(Error{
                .code = ErrorCode::invalid_configuration,
                .message =
                    "OpenAI requires a non-empty API key (set OPENAI_API_KEY or pass api_key), model, base URL, "
                    "and HTTP transport.",
            });
        }
        if (request.messages.empty()) {
            return std::unexpected(Error{
                .code = ErrorCode::invalid_configuration,
                .message = "A generation request must contain at least one message.",
            });
        }

        detail::RequestBody body{
            .model = config_.model,
        };
        if (streaming) {
            body.stream = true;
        }
        if (request.continuation_token) {
            if (request.continuation_token->empty()) {
                return std::unexpected(Error{
                    .code = ErrorCode::invalid_configuration,
                    .message = "A continuation token cannot be empty.",
                });
            }
            body.previous_response_id = request.continuation_token;
        }

        const auto append_input = [&body](const auto& item) -> Result<void> {
            auto encoded_item = to_json(item);
            if (!encoded_item) {
                return std::unexpected(encoded_item.error());
            }
            body.input.emplace_back(std::move(*encoded_item));
            return {};
        };

        body.input.reserve(request.messages.size());
        for (const auto& message : request.messages) {
            if (message.role == MessageRole::tool) {
                if (message.tool_call_id.empty() || !message.tool_calls.empty()) {
                    return std::unexpected(Error{
                        .code = ErrorCode::invalid_tool_call,
                        .message = "A tool result requires a tool call ID and cannot contain tool calls.",
                    });
                }
                auto output = detail::text_content(message);
                if (!output) {
                    return std::unexpected(output.error());
                }
                if (auto result = append_input(detail::FunctionCallOutputInput{
                        .call_id = message.tool_call_id,
                        .output = std::move(*output),
                    });
                    !result) {
                    return std::unexpected(result.error());
                }
                continue;
            }

            const auto role = detail::role_name(message.role);
            if (role.empty() || !message.tool_call_id.empty()) {
                return std::unexpected(Error{
                    .code = ErrorCode::invalid_configuration,
                    .message = "A generation request contains an unsupported role or misplaced tool call ID.",
                });
            }
            if (message.role != MessageRole::assistant && !message.tool_calls.empty()) {
                return std::unexpected(Error{
                    .code = ErrorCode::invalid_tool_call,
                    .message = "Only assistant messages can contain tool calls.",
                });
            }

            if (message.tool_calls.empty() || !message.content.empty()) {
                auto content = detail::input_content(message);
                if (!content) {
                    return std::unexpected(content.error());
                }
                if (auto result = append_input(detail::InputMessage{
                        .role = std::string{role},
                        .content = std::move(*content),
                    });
                    !result) {
                    return std::unexpected(result.error());
                }
            }
            for (const auto& tool_call : message.tool_calls) {
                if (tool_call.id.empty() || tool_call.name.empty() || tool_call.arguments.empty()) {
                    return std::unexpected(Error{
                        .code = ErrorCode::invalid_tool_call,
                        .message = "An assistant tool call requires an ID, name, and JSON arguments.",
                    });
                }
                if (auto result = append_input(detail::FunctionCallInput{
                        .call_id = tool_call.id,
                        .name = tool_call.name,
                        .arguments = tool_call.arguments,
                    });
                    !result) {
                    return std::unexpected(result.error());
                }
            }
        }

        if (!request.tools.empty()) {
            body.tools.emplace();
            body.tools->reserve(request.tools.size());
            for (std::size_t i = 0; i < request.tools.size(); ++i) {
                const auto& tool = request.tools[i];
                if (tool.name.empty()) {
                    return std::unexpected(Error{
                        .code = ErrorCode::invalid_configuration,
                        .message = "Tool names cannot be empty.",
                    });
                }
                for (std::size_t previous = 0; previous < i; ++previous) {
                    if (request.tools[previous].name == tool.name) {
                        return std::unexpected(Error{
                            .code = ErrorCode::invalid_configuration,
                            .message = "Tool names must be unique: " + tool.name,
                        });
                    }
                }
                auto strict_parameters = cail::detail::strict_schema(tool.parameters);
                if (!strict_parameters) {
                    return std::unexpected(strict_parameters.error());
                }
                auto encoded_parameters = to_json(*strict_parameters);
                if (!encoded_parameters) {
                    return std::unexpected(encoded_parameters.error());
                }
                body.tools->push_back(detail::FunctionTool{
                    .name = tool.name,
                    .description = tool.description,
                    .parameters = glz::raw_json{std::move(*encoded_parameters)},
                });
            }
        }

        if (request.structured_output) {
            auto strict_output_schema = cail::detail::strict_schema(request.structured_output->schema);
            if (!strict_output_schema) {
                return std::unexpected(strict_output_schema.error());
            }
            auto encoded_schema = to_json(*strict_output_schema);
            if (!encoded_schema) {
                return std::unexpected(encoded_schema.error());
            }
            body.text = detail::TextOptions{
                .format =
                    detail::JsonSchemaFormat{
                        .name = request.structured_output->name,
                        .description = request.structured_output->description,
                        .schema = glz::raw_json{std::move(*encoded_schema)},
                    },
            };
        }

        auto encoded = to_json(body);
        if (!encoded) {
            return std::unexpected(encoded.error());
        }

        std::string endpoint = config_.base_url;
        while (!endpoint.empty() && endpoint.back() == '/') {
            endpoint.pop_back();
        }
        endpoint += "/responses";

        HttpRequest http_request{
            .url = std::move(endpoint),
            .headers =
                {
                    HttpHeader{.name = "Authorization", .value = "Bearer " + config_.api_key},
                    HttpHeader{.name = "Content-Type", .value = "application/json"},
                    HttpHeader{.name = "Accept", .value = streaming ? "text/event-stream" : "application/json"},
                },
            .body = std::move(*encoded),
        };

        cail::detail::SseParser sse_parser;
        std::optional<detail::ResponseBody> streamed_response;
        std::optional<Error> stream_error;
        std::string reasoning;
        const auto handle_event = [&](const cail::detail::ServerSentEvent& event) {
            if (stop.stop_requested() || stream_error || event.data.empty() || event.data == "[DONE]") {
                return;
            }
            detail::StreamEventBody stream_event{};
            if (const auto error = glz::read<glz::opts{.error_on_unknown_keys = false}>(stream_event, event.data); error) {
                stream_error = Error{
                    .code = ErrorCode::provider_response,
                    .message = glz::format_error(error, event.data),
                    .byte_offset = error.count,
                };
                return;
            }
            if (stream_event.type.empty()) {
                stream_event.type = event.event;
            }
            if (stream_event.type == "response.output_text.delta" && stream_event.delta) {
                on_event(StreamEvent{TextDelta{.text = *stream_event.delta}});
            } else if (stream_event.type == "response.refusal.delta" && stream_event.delta) {
                on_event(StreamEvent{RefusalDelta{.text = *stream_event.delta}});
            } else if ((stream_event.type == "response.reasoning_text.delta" ||
                        stream_event.type == "response.reasoning_summary_text.delta") &&
                       stream_event.delta) {
                reasoning += *stream_event.delta;
                on_event(StreamEvent{ReasoningDelta{.text = *stream_event.delta}});
            } else if (stream_event.type == "response.function_call_arguments.delta" && stream_event.delta) {
                if (stream_event.output_index) {
                    on_event(StreamEvent{ToolCallArgumentsDelta{
                        .output_index = *stream_event.output_index,
                        .arguments = *stream_event.delta,
                    }});
                }
            } else if (stream_event.type == "response.completed" || stream_event.type == "response.incomplete" ||
                       stream_event.type == "response.failed") {
                streamed_response = std::move(stream_event.response);
            } else if (stream_event.type == "error") {
                stream_error = Error{
                    .code = ErrorCode::provider_response,
                    .message = stream_event.message.value_or(
                        stream_event.error ? stream_event.error->message : "OpenAI returned a streaming error."),
                    .provider_code = stream_event.error && stream_event.error->code ? *stream_event.error->code : "",
                    .provider_type = stream_event.error && stream_event.error->type ? *stream_event.error->type : "",
                };
            }
        };

        auto http_response = streaming
                                 ? transport_->stream(http_request, [&](std::string_view bytes) {
                                       sse_parser.feed(bytes, handle_event);
                                   }, stop)
                                 : transport_->send(http_request);
        if (!http_response) {
            return std::unexpected(http_response.error());
        }
        const auto with_context = [&](Error error) {
            error.http_status = http_response->status_code;
            for (const auto& header : http_response->headers) {
                std::string name = header.name;
                std::transform(name.begin(), name.end(), name.begin(), [](unsigned char character) {
                    return static_cast<char>(std::tolower(character));
                });
                if (name == "x-request-id") {
                    error.request_id = header.value;
                    break;
                }
            }
            return std::unexpected(std::move(error));
        };
        if (stop.stop_requested()) {
            return with_context(Error{.code = ErrorCode::cancelled, .message = "Generation was cancelled."});
        }
        if (http_response->status_code < 200 || http_response->status_code >= 300) {
            detail::ErrorBody error_body{};
            const auto parse_error = glz::read<glz::opts{.error_on_unknown_keys = false}>(
                error_body, http_response->body);
            const auto message = !parse_error && error_body.error ? error_body.error->message : http_response->body;
            return with_context(Error{
                .code = ErrorCode::http_status,
                .message = "OpenAI returned HTTP " + std::to_string(http_response->status_code) +
                           (message.empty() ? "." : ": " + message),
                .provider_code = !parse_error && error_body.error && error_body.error->code
                                     ? *error_body.error->code
                                     : "",
                .provider_type = !parse_error && error_body.error && error_body.error->type
                                     ? *error_body.error->type
                                     : "",
            });
        }

        if (streaming) {
            sse_parser.finish(handle_event);
            if (stream_error) {
                return with_context(std::move(*stream_error));
            }
            if (!streamed_response) {
                return with_context(Error{
                    .code = ErrorCode::provider_response,
                    .message = "OpenAI closed the event stream without a completed response.",
                    .http_status = http_response->status_code,
                });
            }
            std::vector<std::size_t> call_indexes;
            for (std::size_t index = 0; index < streamed_response->output.size(); ++index) {
                if (streamed_response->output[index].type == "function_call") {
                    call_indexes.push_back(index);
                }
            }
            auto result = detail::decode_response(std::move(*streamed_response), http_response->status_code);
            if (!result) {
                return with_context(result.error());
            }
            if (result->reasoning.empty()) {
                result->reasoning = std::move(reasoning);
            }
            for (std::size_t index = 0; index < result->tool_calls.size(); ++index) {
                on_event(StreamEvent{ToolCallReady{
                    .output_index = call_indexes[index],
                    .call = result->tool_calls[index],
                }});
            }
            if (result->usage) {
                on_event(StreamEvent{UsageUpdate{.usage = *result->usage}});
            }
            return result;
        }

        detail::ResponseBody response_body{};
        if (const auto error = glz::read<glz::opts{.error_on_unknown_keys = false}>(
                response_body, http_response->body);
            error) {
            return with_context(Error{
                .code = ErrorCode::provider_response,
                .message = glz::format_error(error, http_response->body),
                .byte_offset = error.count,
                .http_status = http_response->status_code,
            });
        }

        auto result = detail::decode_response(std::move(response_body), http_response->status_code);
        if (!result) {
            return with_context(result.error());
        }
        return result;
    }

    public:
    [[nodiscard]] Result<GenerationResponse> generate(std::string_view prompt) const {
        return generate(GenerationRequest{
            .messages = {Message{.role = MessageRole::user, .content = {TextPart{.text = std::string{prompt}}}}},
        });
    }

    [[nodiscard]] Result<GenerationResponse> generate(
        const GenerationRequest& request,
        std::vector<Tool> tools,
        ToolLoopOptions options = {}) const
    {
        return cail::run_tool_loop(*this, request, tools, options);
    }

    [[nodiscard]] Result<GenerationResponse> generate(
        std::string_view prompt,
        std::vector<Tool> tools,
        ToolLoopOptions options = {}) const
    {
        return generate(
            GenerationRequest{
                .messages = {Message{.role = MessageRole::user, .content = {TextPart{.text = std::string{prompt}}}}},
            },
            std::move(tools),
            options);
    }

    template <typename T> [[nodiscard]] Result<T> generate(const GenerationRequest& request) const {
        GenerationRequest structured_request = request;
        structured_request.structured_output = StructuredOutput{
            .name = "cail_output",
            .schema = cail::schema<T>(),
        };

        auto response = generate(structured_request);
        if (!response) {
            return std::unexpected(response.error());
        }
        if (response->status == GenerationStatus::refused) {
            return std::unexpected(Error{
                .code = ErrorCode::refused,
                .message = response->text.empty() ? "OpenAI refused the request." : response->text,
            });
        }
        if (response->status == GenerationStatus::incomplete) {
            return std::unexpected(Error{
                .code = ErrorCode::incomplete_response,
                .message = "OpenAI returned an incomplete structured response.",
            });
        }
        if (!response->tool_calls.empty()) {
            return std::unexpected(Error{
                .code = ErrorCode::tool_call_required,
                .message = "OpenAI requested a tool call before returning structured output.",
            });
        }
        return from_json<T>(response->text);
    }

    template <typename T> [[nodiscard]] Result<T> generate(std::string_view prompt) const {
        return generate<T>(GenerationRequest{
            .messages = {Message{.role = MessageRole::user, .content = {TextPart{.text = std::string{prompt}}}}},
        });
    }

    private:
    Config config_;
    std::unique_ptr<HttpTransport> transport_;
};

} // namespace cail::detail::openai

namespace cail {

struct OpenAIProviderSettings {
    std::string api_key;
    std::string base_url{"https://api.openai.com/v1"};
};

class OpenAIProvider {
    public:
    explicit OpenAIProvider(OpenAIProviderSettings settings = {}) : settings_(std::move(settings)) {}

    [[nodiscard]] LanguageModel operator()(std::string model_id) const
    {
        auto api_key = settings_.api_key;
        if (api_key.empty()) {
            if (const char* env = std::getenv("OPENAI_API_KEY"); env != nullptr && env[0] != '\0') {
                api_key = env;
            }
        }
        auto client = std::make_shared<detail::openai::Client>(detail::openai::Config{
            .api_key = std::move(api_key),
            .model = std::move(model_id),
            .base_url = settings_.base_url,
        });
        return LanguageModel{
            [client](const GenerationRequest& request) { return client->generate(request); },
            [client](const GenerationRequest& request, const StreamHandler& on_event, std::stop_token stop) {
                return client->stream(request, on_event, stop);
            }, LanguageModelCapabilities{
                .image_input = true,
                .tools = true,
                .structured_output = true,
                .reasoning = true,
                .continuation = true,
            }};
    }

    [[nodiscard]] EmbeddingModel embedding_model(
        std::string model_id, std::optional<std::size_t> dimensions = std::nullopt) const
    {
        auto api_key = settings_.api_key;
        if (api_key.empty()) {
            if (const char* env = std::getenv("OPENAI_API_KEY"); env != nullptr && env[0] != '\0') {
                api_key = env;
            }
        }
        auto client = std::make_shared<detail::openai::EmbeddingClient>(
            std::move(api_key), std::move(model_id), settings_.base_url, dimensions);
        return EmbeddingModel{[client](const std::vector<std::string>& inputs) {
            return client->embed_many(inputs);
        }};
    }

    private:
    OpenAIProviderSettings settings_;
};

[[nodiscard]] inline OpenAIProvider create_openai(OpenAIProviderSettings settings = {})
{
    return OpenAIProvider{std::move(settings)};
}

inline OpenAIProvider openai{};

} // namespace cail
