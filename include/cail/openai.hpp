#pragma once

#include <cail/detail/base64.hpp>
#include <cail/detail/env.hpp>
#include <cail/detail/glaze_http_transport.hpp>
#include <cail/detail/http_context.hpp>
#include <cail/detail/request_headers.hpp>
#include <cail/detail/sse.hpp>
#include <cail/detail/strict_schema.hpp>
#include <cail/generation.hpp>
#include <cail/json.hpp>
#include <cail/language_model.hpp>
#include <cail/openai_embeddings.hpp>
#include <cail/detail/openai_responses_wire.hpp>
#include <cail/tool.hpp>

#include <glaze/glaze.hpp>

#include <algorithm>
#include <exception>
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
  std::string request_session_header;
  bool prompt_cache_key = false;
};

struct MessageOptions {
  std::vector<glz::raw_json> reasoning_details;
};

class Client {
public:
  explicit Client(Config config)
      : Client(std::move(config),
               std::make_unique<cail::detail::GlazeHttpTransport>()) {}

  Client(Config config, std::unique_ptr<HttpTransport> transport)
      : config_(std::move(config)), transport_(std::move(transport)) {}

  [[nodiscard]] Result<GenerationResponse>
  generate(const GenerationRequest &request) const {
    return generate_impl(request, {});
  }

  [[nodiscard]] Result<GenerationResponse>
  stream(const GenerationRequest &request, const StreamHandler &on_event,
         std::stop_token stop = {}) const {
    return generate_impl(request, on_event, stop);
  }

  [[nodiscard]] Result<GenerationResponse>
  stream(std::string_view prompt, const StreamHandler &on_event,
         std::stop_token stop = {}) const {
    return stream(
        GenerationRequest{
            .messages = {Message{
                .role = MessageRole::user,
                .content = {TextPart{.text = std::string{prompt}}}}},
        },
        on_event, stop);
  }

private:
  [[nodiscard]] Result<GenerationResponse>
  generate_impl(const GenerationRequest &request, const StreamHandler &on_event,
                std::stop_token stop = {}) const {
    const bool streaming = static_cast<bool>(on_event);
    if (stop.stop_requested()) {
      return std::unexpected(generation_cancelled_error());
    }
    if (auto valid = cail::detail::validate_max_output_tokens(request); !valid) {
      return std::unexpected(valid.error());
    }
    if (config_.api_key.empty() || config_.model.empty() ||
        config_.base_url.empty() || !transport_) {
      return std::unexpected(Error{
          .code = ErrorCode::invalid_configuration,
          .message =
              "The OpenAI Responses adapter requires a non-empty API key "
              "(set the provider's environment key or pass api_key), model, "
              "base URL, "
              "and HTTP transport.",
      });
    }
    if (request.messages.empty()) {
      return std::unexpected(Error{
          .code = ErrorCode::invalid_configuration,
          .message = "A generation request must contain at least one message.",
      });
    }

    wire::RequestBody body{
        .model = config_.model,
        .max_output_tokens = request.max_output_tokens,
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

    const auto append_input = [&body](const auto &item) -> Result<void> {
      auto encoded_item = to_json(item);
      if (!encoded_item) {
        return std::unexpected(encoded_item.error());
      }
      body.input.emplace_back(std::move(*encoded_item));
      return {};
    };

    std::vector<ContentPart> pending_tool_images;
    const auto flush_tool_images = [&]() -> Result<void> {
      if (pending_tool_images.empty()) {
        return {};
      }
      auto content = wire::input_content(
          Message{.role = MessageRole::user,
                  .content = std::move(pending_tool_images)});
      if (!content) {
        return std::unexpected(content.error());
      }
      pending_tool_images.clear();
      return append_input(wire::InputMessage{
          .role = "user", .content = std::move(*content)});
    };

    body.input.reserve(request.messages.size());
    for (const auto &message : request.messages) {
      if (message.role != MessageRole::tool) {
        if (auto flushed = flush_tool_images(); !flushed) {
          return std::unexpected(flushed.error());
        }
      }
      if (message.role == MessageRole::tool) {
        if (message.tool_call_id.empty() || !message.tool_calls.empty()) {
          return std::unexpected(Error{
              .code = ErrorCode::invalid_tool_call,
              .message = "A tool result requires a tool call ID and cannot "
                         "contain tool calls.",
          });
        }
        std::string output;
        for (const auto &part : message.content) {
          if (const auto *text = std::get_if<TextPart>(&part)) {
            output += text->text;
          } else {
            const auto &image = std::get<ImagePart>(part);
            if (image.mime_type.empty() || image.bytes.empty()) {
              return std::unexpected(Error{
                  .code = ErrorCode::invalid_configuration,
                  .message =
                      "Tool result images require non-empty bytes and a MIME "
                      "type.",
              });
            }
            pending_tool_images.emplace_back(image);
          }
        }
        if (auto result = append_input(wire::FunctionCallOutputInput{
                .call_id = message.tool_call_id,
                .output = std::move(output),
            });
            !result) {
          return std::unexpected(result.error());
        }
        continue;
      }

      const auto role = wire::role_name(message.role);
      if (role.empty() || !message.tool_call_id.empty()) {
        return std::unexpected(Error{
            .code = ErrorCode::invalid_configuration,
            .message = "A generation request contains an unsupported role or "
                       "misplaced tool call ID.",
        });
      }
      if (message.role != MessageRole::assistant &&
          !message.tool_calls.empty()) {
        return std::unexpected(Error{
            .code = ErrorCode::invalid_tool_call,
            .message = "Only assistant messages can contain tool calls.",
        });
      }

      if (message.role == MessageRole::assistant && message.provider_options) {
        MessageOptions options;
        if (const auto error =
                glz::read<glz::opts{.error_on_unknown_keys = false}>(
                    options, *message.provider_options);
            error) {
          return std::unexpected(Error{
              .code = ErrorCode::invalid_configuration,
              .message = "Assistant provider options must be a JSON object: " +
                         glz::format_error(error, *message.provider_options),
              .byte_offset = error.count,
          });
        }
        for (const auto &raw_item : options.reasoning_details) {
          auto item = wire::decode_response_item(raw_item, 0);
          if (!item) {
            return std::unexpected(item.error());
          }
          if (item->type == "reasoning" && item->encrypted_content) {
            body.input.emplace_back(raw_item);
          }
        }
      }

      if (message.tool_calls.empty() || !message.content.empty()) {
        auto content = wire::input_content(message);
        if (!content) {
          return std::unexpected(content.error());
        }
        if (auto result = append_input(wire::InputMessage{
                .role = std::string{role},
                .content = std::move(*content),
            });
            !result) {
          return std::unexpected(result.error());
        }
      }
      for (const auto &tool_call : message.tool_calls) {
        if (tool_call.id.empty() || tool_call.name.empty() ||
            tool_call.arguments.empty()) {
          return std::unexpected(Error{
              .code = ErrorCode::invalid_tool_call,
              .message = "An assistant tool call requires an ID, name, and "
                         "JSON arguments.",
          });
        }
        if (auto result = append_input(wire::FunctionCallInput{
                .call_id = tool_call.id,
                .name = tool_call.name,
                .arguments = tool_call.arguments,
            });
            !result) {
          return std::unexpected(result.error());
        }
      }
    }
    if (auto flushed = flush_tool_images(); !flushed) {
      return std::unexpected(flushed.error());
    }

    if (!request.tools.empty()) {
      body.tools.emplace();
      body.tools->reserve(request.tools.size());
      for (std::size_t i = 0; i < request.tools.size(); ++i) {
        const auto &tool = request.tools[i];
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
        auto strict_parameters =
            cail::detail::strict_json_schema(tool.parameters);
        if (!strict_parameters) {
          return std::unexpected(strict_parameters.error());
        }
        auto encoded_parameters = to_json(*strict_parameters);
        if (!encoded_parameters) {
          return std::unexpected(encoded_parameters.error());
        }
        body.tools->push_back(wire::FunctionTool{
            .name = tool.name,
            .description = tool.description,
            .parameters = glz::raw_json{std::move(*encoded_parameters)},
        });
      }
    }

    if (request.structured_output) {
      auto strict_output_schema =
          cail::detail::strict_json_schema(request.structured_output->schema);
      if (!strict_output_schema) {
        return std::unexpected(strict_output_schema.error());
      }
      auto encoded_schema = to_json(*strict_output_schema);
      if (!encoded_schema) {
        return std::unexpected(encoded_schema.error());
      }
      body.text = wire::TextOptions{
          .format =
              wire::JsonSchemaFormat{
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
    if (config_.prompt_cache_key && !request.session_id.empty()) {
      glz::generic cache_key = glz::generic::object_t{};
      cache_key["prompt_cache_key"] = request.session_id;
      auto dumped = cache_key.dump();
      if (!dumped) {
        return std::unexpected(Error{
            .code = ErrorCode::json_serialization,
            .message = "Could not encode the OpenAI prompt cache key.",
        });
      }
      auto merged = merge_json_objects(*encoded, *dumped);
      if (!merged) {
        return std::unexpected(merged.error());
      }
      encoded = std::move(*merged);
    }
    if (request.provider_options) {
      auto options = *request.provider_options;
      glz::generic parsed;
      if (!glz::read_json(parsed, options) && parsed.is_object()) {
        auto& object = parsed.get<glz::generic::object_t>();
        if (const auto it = object.find("reasoning_effort");
            it != object.end() && it->second.is_string()) {
          const auto effort = it->second.get<std::string>();
          object.erase(it);
          if (!parsed.contains("reasoning") || !parsed["reasoning"].is_object()) {
            parsed["reasoning"] = glz::generic::object_t{};
          }
          parsed["reasoning"]["effort"] = effort;
          if (auto dumped = parsed.dump()) {
            options = std::move(*dumped);
          }
        }
      }
      auto merged = merge_json_objects(*encoded, options);
      if (!merged) {
        return std::unexpected(merged.error());
      }
      encoded = std::move(*merged);
    }

    std::string endpoint = config_.base_url;
    while (!endpoint.empty() && endpoint.back() == '/') {
      endpoint.pop_back();
    }
    if (endpoint.find("/responses") == std::string::npos) {
      endpoint += "/responses";
    }

    HttpRequest http_request{
        .url = std::move(endpoint),
        .headers =
            {
                HttpHeader{.name = "Authorization",
                           .value = "Bearer " + config_.api_key},
                HttpHeader{.name = "Content-Type", .value = "application/json"},
                HttpHeader{.name = "Accept",
                           .value = streaming ? "text/event-stream"
                                              : "application/json"},
            },
        .body = std::move(*encoded),
    };
    cail::detail::append_session_header(http_request.headers,
                                        config_.request_session_header,
                                        request.session_id);
    if (request.before_request) {
      try {
        request.before_request(http_request);
      } catch (const std::exception& error) {
        return std::unexpected(Error{
            .code = ErrorCode::invalid_configuration,
            .message = std::string{"The before-request callback failed: "} +
                       error.what()});
      } catch (...) {
        return std::unexpected(Error{
            .code = ErrorCode::invalid_configuration,
            .message = "The before-request callback failed."});
      }
    }

    cail::detail::SseParser sse_parser;
    std::optional<wire::ResponseBody> streamed_response;
    std::optional<Error> stream_error;
    std::string reasoning;
    const auto handle_event = [&](const cail::detail::ServerSentEvent &event) {
      if (stop.stop_requested() || stream_error || event.data.empty() ||
          event.data == "[DONE]") {
        return;
      }
      wire::StreamEventBody stream_event{};
      if (const auto error =
              glz::read<glz::opts{.error_on_unknown_keys = false}>(stream_event,
                                                                   event.data);
          error) {
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
      if (stream_event.type == "response.output_text.delta" &&
          stream_event.delta) {
        on_event(StreamEvent{TextDelta{.text = *stream_event.delta}});
      } else if (stream_event.type == "response.refusal.delta" &&
                 stream_event.delta) {
        on_event(StreamEvent{RefusalDelta{.text = *stream_event.delta}});
      } else if ((stream_event.type == "response.reasoning_text.delta" ||
                  stream_event.type ==
                      "response.reasoning_summary_text.delta") &&
                 stream_event.delta) {
        reasoning += *stream_event.delta;
        on_event(StreamEvent{ReasoningDelta{.text = *stream_event.delta}});
      } else if (stream_event.type ==
                     "response.function_call_arguments.delta" &&
                 stream_event.delta) {
        if (stream_event.output_index) {
          on_event(StreamEvent{ToolCallArgumentsDelta{
              .output_index = *stream_event.output_index,
              .arguments = *stream_event.delta,
          }});
        }
      } else if (stream_event.type == "response.completed" ||
                 stream_event.type == "response.incomplete" ||
                 stream_event.type == "response.failed") {
        streamed_response = std::move(stream_event.response);
      } else if (stream_event.type == "error") {
        stream_error = Error{
            .code = ErrorCode::provider_response,
            .message = stream_event.message.value_or(
                stream_event.error ? stream_event.error->message
                                   : "OpenAI returned a streaming error."),
            .provider_code = stream_event.error && stream_event.error->code
                                 ? *stream_event.error->code
                                 : "",
            .provider_type = stream_event.error && stream_event.error->type
                                 ? *stream_event.error->type
                                 : "",
        };
      }
    };

    auto http_response = streaming ? transport_->stream(
                                         http_request,
                                         [&](std::string_view bytes) {
                                           sse_parser.feed(bytes, handle_event);
                                         },
                                         stop)
                                   : transport_->send(http_request);
    if (!http_response) {
      return std::unexpected(http_response.error());
    }
    if (request.after_response) {
      try {
        request.after_response(*http_response);
      } catch (const std::exception& error) {
        return std::unexpected(Error{
            .code = ErrorCode::provider_response,
            .message = std::string{"The after-response callback failed: "} +
                       error.what()});
      } catch (...) {
        return std::unexpected(Error{
            .code = ErrorCode::provider_response,
            .message = "The after-response callback failed."});
      }
    }
    const auto with_context = [&](Error error) {
      return unexpected_with_http_context<GenerationResponse>(std::move(error),
                                                              *http_response);
    };
    if (stop.stop_requested()) {
      return with_context(generation_cancelled_error());
    }
    if (http_response->status_code < 200 || http_response->status_code >= 300) {
      wire::ErrorBody error_body{};
      const auto parse_error =
          glz::read<glz::opts{.error_on_unknown_keys = false}>(
              error_body, http_response->body);
      const auto message = !parse_error && error_body.error
                               ? error_body.error->message
                               : http_response->body;
      return with_context(Error{
          .code = ErrorCode::http_status,
          .message = "OpenAI returned HTTP " +
                     std::to_string(http_response->status_code) +
                     (message.empty() ? "." : ": " + message),
          .provider_code =
              !parse_error && error_body.error && error_body.error->code
                  ? *error_body.error->code
                  : "",
          .provider_type =
              !parse_error && error_body.error && error_body.error->type
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
            .message =
                "OpenAI closed the event stream without a completed response.",
            .http_status = http_response->status_code,
        });
      }
      std::vector<std::size_t> call_indexes;
      for (std::size_t index = 0; index < streamed_response->output.size();
           ++index) {
        auto item = wire::decode_response_item(streamed_response->output[index],
                                               http_response->status_code);
        if (!item) {
          return with_context(item.error());
        }
        if (item->type == "function_call") {
          call_indexes.push_back(index);
        }
      }
      auto result = wire::decode_response(std::move(*streamed_response),
                                          http_response->status_code);
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

    wire::ResponseBody response_body{};
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

    auto result = wire::decode_response(std::move(response_body),
                                        http_response->status_code);
    if (!result) {
      return with_context(result.error());
    }
    return result;
  }

public:
  [[nodiscard]] Result<GenerationResponse>
  generate(std::string_view prompt) const {
    return generate(GenerationRequest{
        .messages = {Message{
            .role = MessageRole::user,
            .content = {TextPart{.text = std::string{prompt}}}}},
    });
  }

  [[nodiscard]] Result<GenerationResponse>
  generate(const GenerationRequest &request, std::vector<Tool> tools,
           ToolLoopOptions options = {}) const {
    return cail::run_tool_loop(*this, request, tools, options);
  }

  [[nodiscard]] Result<GenerationResponse>
  generate(std::string_view prompt, std::vector<Tool> tools,
           ToolLoopOptions options = {}) const {
    return generate(
        GenerationRequest{
            .messages = {Message{
                .role = MessageRole::user,
                .content = {TextPart{.text = std::string{prompt}}}}},
        },
        std::move(tools), options);
  }

  template <typename T>
  [[nodiscard]] Result<T> generate(const GenerationRequest &request) const {
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
          .message = response->text.empty() ? "OpenAI refused the request."
                                            : response->text,
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
          .message = "OpenAI requested a tool call before returning structured "
                     "output.",
      });
    }
    return from_json<T>(response->text);
  }

  template <typename T>
  [[nodiscard]] Result<T> generate(std::string_view prompt) const {
    return generate<T>(GenerationRequest{
        .messages = {Message{
            .role = MessageRole::user,
            .content = {TextPart{.text = std::string{prompt}}}}},
    });
  }

private:
  Config config_;
  std::unique_ptr<HttpTransport> transport_;
};

} // namespace cail::detail::openai

namespace cail::detail {

[[nodiscard]] constexpr AdapterCapabilities openai_responses_adapter_capabilities()
{
    return AdapterCapabilities{
        .image_input = true,
        .tools = true,
        .structured_output = true,
        .reasoning = true,
        .continuation = true,
    };
}

[[nodiscard]] inline std::shared_ptr<openai::Client>
make_openai_responses_client(openai::Config config,
                             std::unique_ptr<HttpTransport> transport = {})
{
    if (transport) {
        return std::make_shared<openai::Client>(std::move(config), std::move(transport));
    }
    return std::make_shared<openai::Client>(std::move(config));
}

[[nodiscard]] inline LanguageModel
language_model_from(const std::shared_ptr<openai::Client>& client)
{
    return LanguageModel{
        [client](const GenerationRequest& request) { return client->generate(request); },
        [client](const GenerationRequest& request, const StreamHandler& handler, std::stop_token stop) {
            return client->stream(request, handler, stop);
        },
        openai_responses_adapter_capabilities(),
    };
}

} // namespace cail::detail

#include <cail/detail/openai_responses_preset.hpp>

namespace cail {

struct OpenAITag {
    static constexpr const char* env_var = "OPENAI_API_KEY";
    static constexpr const char* default_base_url = "https://api.openai.com/v1";
};

using OpenAIProviderSettings = OpenAiResponsesPresetSettings<OpenAITag>;

class OpenAIProvider {
    public:
    explicit OpenAIProvider(OpenAIProviderSettings settings = {})
        : settings_(settings), preset_(std::move(settings)) {}

    [[nodiscard]] LanguageModel operator()(std::string model_id) const
    {
        return preset_(std::move(model_id));
    }

    [[nodiscard]] LanguageModel operator()(std::string model_id,
                                           std::unique_ptr<HttpTransport> transport) const
    {
        return preset_(std::move(model_id), std::move(transport));
    }

    [[nodiscard]] EmbeddingModel
    embedding_model(std::string model_id,
                    std::optional<std::size_t> dimensions = std::nullopt) const
    {
        auto api_key = detail::env_or(settings_.api_key, OpenAITag::env_var);
        auto client = std::make_shared<detail::openai::EmbeddingClient>(
            std::move(api_key), std::move(model_id), settings_.base_url, dimensions);
        return EmbeddingModel{[client](const std::vector<std::string>& inputs) {
            return client->embed_many(inputs);
        }};
    }

    private:
    OpenAIProviderSettings settings_;
    detail::OpenAiResponsesPresetProvider<OpenAITag> preset_;
};

[[nodiscard]] inline OpenAIProvider create_openai(OpenAIProviderSettings settings = {})
{
    return OpenAIProvider{std::move(settings)};
}

inline OpenAIProvider openai{};

} // namespace cail
