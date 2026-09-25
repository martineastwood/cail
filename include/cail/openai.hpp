#pragma once

#include <cail/detail/glaze_http_transport.hpp>
#include <cail/detail/sse.hpp>
#include <cail/generation.hpp>
#include <cail/language_model.hpp>
#include <cail/json.hpp>
#include <cail/tool.hpp>

#include <glaze/glaze.hpp>

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <optional>
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
    std::string content;
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
    std::optional<std::string> call_id;
    std::optional<std::string> name;
    std::optional<std::string> arguments;
};

struct ResponseUsage {
    std::size_t input_tokens{};
    std::size_t output_tokens{};
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

[[nodiscard]] inline Result<void> validate_strict_schema(const Schema& schema) {
    if (schema.type == SchemaType::object) {
        if (!schema.properties || !schema.required || !schema.additional_properties || *schema.additional_properties) {
            return std::unexpected(Error{
                .code = ErrorCode::unsupported_schema,
                .message = "OpenAI strict structured output requires object properties, a required list, and "
                           "additionalProperties=false.",
            });
        }

        for (const auto& [name, child] : *schema.properties) {
            if (std::ranges::find(*schema.required, name) == schema.required->end()) {
                return std::unexpected(Error{
                    .code = ErrorCode::unsupported_schema,
                    .message = "OpenAI strict structured output requires every property to be required. Optional "
                               "nullable fields are not supported yet: " +
                               name,
                });
            }
            if (!child) {
                return std::unexpected(Error{
                    .code = ErrorCode::unsupported_schema,
                    .message = "The structured output schema contains a null property schema.",
                });
            }
            if (auto result = validate_strict_schema(*child); !result) {
                return result;
            }
        }
    } else if (schema.type == SchemaType::array) {
        if (!schema.items) {
            return std::unexpected(Error{
                .code = ErrorCode::unsupported_schema,
                .message = "OpenAI structured output arrays require an items schema.",
            });
        }
        return validate_strict_schema(*schema.items);
    }

    return {};
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
        const GenerationRequest& request, const TextDeltaHandler& on_text_delta) const {
        if (!on_text_delta) {
            return std::unexpected(Error{
                .code = ErrorCode::invalid_configuration,
                .message = "Streaming requires a text delta handler.",
            });
        }
        return generate_impl(request, on_text_delta);
    }

    [[nodiscard]] Result<GenerationResponse> stream(std::string_view prompt, const TextDeltaHandler& on_text_delta) const {
        return stream(GenerationRequest{
            .messages = {Message{.role = MessageRole::user, .content = std::string{prompt}}},
        }, on_text_delta);
    }

    private:
    [[nodiscard]] Result<GenerationResponse> generate_impl(
        const GenerationRequest& request, const TextDeltaHandler& on_text_delta) const {
        const bool streaming = static_cast<bool>(on_text_delta);
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
                if (auto result = append_input(detail::FunctionCallOutputInput{
                        .call_id = message.tool_call_id,
                        .output = message.content,
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
                if (auto result = append_input(detail::InputMessage{
                        .role = std::string{role},
                        .content = message.content,
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
                if (auto result = detail::validate_strict_schema(tool.parameters); !result) {
                    return std::unexpected(result.error());
                }
                auto encoded_parameters = to_json(tool.parameters);
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
            if (auto result = detail::validate_strict_schema(request.structured_output->schema); !result) {
                return std::unexpected(result.error());
            }
            auto encoded_schema = to_json(request.structured_output->schema);
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
        const auto handle_event = [&](const cail::detail::ServerSentEvent& event) {
            if (stream_error || event.data.empty() || event.data == "[DONE]") {
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
            if ((stream_event.type == "response.output_text.delta" || stream_event.type == "response.refusal.delta") &&
                stream_event.delta) {
                on_text_delta(*stream_event.delta);
            } else if (stream_event.type == "response.completed" || stream_event.type == "response.incomplete" ||
                       stream_event.type == "response.failed") {
                streamed_response = std::move(stream_event.response);
            } else if (stream_event.type == "error") {
                stream_error = Error{
                    .code = ErrorCode::provider_response,
                    .message = stream_event.message.value_or(
                        stream_event.error ? stream_event.error->message : "OpenAI returned a streaming error."),
                };
            }
        };

        auto http_response = streaming
                                 ? transport_->stream(http_request, [&](std::string_view bytes) {
                                       sse_parser.feed(bytes, handle_event);
                                   })
                                 : transport_->send(http_request);
        if (!http_response) {
            return std::unexpected(http_response.error());
        }
        if (http_response->status_code < 200 || http_response->status_code >= 300) {
            detail::ErrorBody error_body{};
            const auto parse_error = glz::read<glz::opts{.error_on_unknown_keys = false}>(
                error_body, http_response->body);
            const auto message = !parse_error && error_body.error ? error_body.error->message : http_response->body;
            return std::unexpected(Error{
                .code = ErrorCode::http_status,
                .message = "OpenAI returned HTTP " + std::to_string(http_response->status_code) +
                           (message.empty() ? "." : ": " + message),
                .http_status = http_response->status_code,
            });
        }

        if (streaming) {
            sse_parser.finish(handle_event);
            if (stream_error) {
                stream_error->http_status = http_response->status_code;
                return std::unexpected(std::move(*stream_error));
            }
            if (!streamed_response) {
                return std::unexpected(Error{
                    .code = ErrorCode::provider_response,
                    .message = "OpenAI closed the event stream without a completed response.",
                    .http_status = http_response->status_code,
                });
            }
            return detail::decode_response(std::move(*streamed_response), http_response->status_code);
        }

        detail::ResponseBody response_body{};
        if (const auto error = glz::read<glz::opts{.error_on_unknown_keys = false}>(
                response_body, http_response->body);
            error) {
            return std::unexpected(Error{
                .code = ErrorCode::provider_response,
                .message = glz::format_error(error, http_response->body),
                .byte_offset = error.count,
                .http_status = http_response->status_code,
            });
        }

        return detail::decode_response(std::move(response_body), http_response->status_code);
    }

    public:
    [[nodiscard]] Result<GenerationResponse> generate(std::string_view prompt) const {
        return generate(GenerationRequest{
            .messages = {Message{.role = MessageRole::user, .content = std::string{prompt}}},
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
                .messages = {Message{.role = MessageRole::user, .content = std::string{prompt}}},
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
            .messages = {Message{.role = MessageRole::user, .content = std::string{prompt}}},
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
            [client](const GenerationRequest& request, const TextDeltaHandler& on_text_delta) {
                return client->stream(request, on_text_delta);
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
