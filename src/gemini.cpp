#include "detail/gemini.hpp"
#include "detail/async_http.hpp"
#include "detail/encode_json.hpp"
#include "detail/http_context.hpp"
#include "detail/request_headers.hpp"
#include "detail/sse.hpp"
#include <cail/detail/base64.hpp>
#include <cail/detail/env.hpp>
#include <cail/embedding_model.hpp>
#include <cail/gemini.hpp>
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

namespace cail::detail::gemini {

const cail::ToolCall* find_tool_call(const GenerationRequest& request, std::string_view id) {
  for (auto it = request.messages.rbegin(); it != request.messages.rend(); ++it) {
    for (const auto& call : it->tool_calls) {
      if (call.id == id) {
        return &call;
      }
    }
  }
  return nullptr;
}

Result<RequestBody> encode(const GenerationRequest& request) {
  if (auto valid = cail::detail::validate_pdf_parts(request); !valid) {
    return std::unexpected(valid.error());
  }
  if (auto valid = cail::detail::validate_request_controls(request, 2.0, 5); !valid) {
    return std::unexpected(valid.error());
  }
  if (auto valid = cail::detail::validate_max_output_tokens(request); !valid) {
    return std::unexpected(valid.error());
  }
  if (request.messages.empty() || request.continuation_token) {
    return std::unexpected(
        Error{.code = ErrorCode::invalid_configuration,
              .message = "Gemini requires messages and does not accept continuation tokens."});
  }
  RequestBody body;
  if (request.max_output_tokens || request.temperature || request.top_p ||
      !request.stop_sequences.empty()) {
    body.generationConfig = GenerationConfig{.maxOutputTokens = request.max_output_tokens,
                                             .temperature = request.temperature,
                                             .topP = request.top_p};
    if (!request.stop_sequences.empty()) {
      body.generationConfig->stopSequences = request.stop_sequences;
    }
  }
  if (request.tool_choice) {
    const auto& choice = *request.tool_choice;
    body.toolConfig = RequestBody::ToolConfig{
        .functionCallingConfig = {.mode = choice.mode == ToolChoiceMode::none    ? "NONE"
                                          : choice.mode == ToolChoiceMode::auto_ ? "AUTO"
                                                                                 : "ANY"}};
    if (choice.mode == ToolChoiceMode::named) {
      body.toolConfig->functionCallingConfig.allowedFunctionNames =
          std::vector<std::string>{choice.name};
    }
  }
  for (const auto& message : request.messages) {
    if (message.role == MessageRole::developer) {
      return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                   .message = "Gemini does not accept developer-role messages."});
    }
    if (message.role == MessageRole::system) {
      if (!body.contents.empty() || body.systemInstruction || !message.tool_calls.empty()) {
        return std::unexpected(
            Error{.code = ErrorCode::invalid_configuration,
                  .message = "Gemini requires one system message before conversation messages."});
      }
      Content system{.role = "system"};
      for (const auto& part : message.content) {
        const auto* value = std::get_if<cail::TextPart>(&part);
        if (!value) {
          return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                       .message = "Gemini system instructions must contain text."});
        }
        if (auto added = append_json(system.parts, TextPart{.text = value->text}); !added) {
          return std::unexpected(added.error());
        }
      }
      body.systemInstruction = std::move(system);
      continue;
    }
    if (message.role == MessageRole::tool) {
      if (body.contents.empty() || !message.tool_calls.empty()) {
        return std::unexpected(Error{.code = ErrorCode::invalid_tool_call,
                                     .message = "Gemini tool results require a preceding model "
                                                "call and cannot contain tool calls."});
      }
      const auto* matched = find_tool_call(request, message.tool_call_id);
      std::string output;
      for (const auto& part : message.content) {
        if (const auto* text = std::get_if<cail::TextPart>(&part)) {
          output += text->text;
        }
      }
      if (!matched || matched->id.empty()) {
        return std::unexpected(Error{.code = ErrorCode::invalid_tool_call,
                                     .message = "Gemini tool result requires a known call ID."});
      }
      glz::generic value;
      if (glz::read_json(value, output) || !value.is_object()) {
        auto wrapped = to_json(std::map<std::string, std::string>{{"output", output}});
        if (!wrapped) {
          return std::unexpected(wrapped.error());
        }
        output = std::move(*wrapped);
      }
      auto& parts = body.contents.emplace_back(Content{.role = "user"}).parts;
      auto added = append_json(
          parts, ResultPart{.functionResponse = FunctionResponse{.name = matched->name,
                                                                 .response = glz::raw_json{output},
                                                                 .id = matched->id}});
      if (!added) {
        return std::unexpected(added.error());
      }
      for (const auto& part : message.content) {
        const auto* image = std::get_if<cail::ImagePart>(&part);
        if (!image) {
          continue;
        }
        if (image->mime_type.empty() || image->bytes.empty()) {
          return std::unexpected(
              Error{.code = ErrorCode::invalid_configuration,
                    .message = "Gemini tool result images require bytes and a MIME type."});
        }
        added = append_json(
            parts,
            ImagePart{.inlineData = InlineData{.mimeType = image->mime_type,
                                               .data = cail::detail::base64_encode(image->bytes)}});
        if (!added) {
          return std::unexpected(added.error());
        }
      }
      continue;
    }
    Content content{.role = message.role == MessageRole::assistant ? "model" : "user"};
    for (const auto& part : message.content) {
      Result<void> added;
      if (const auto* value = std::get_if<cail::TextPart>(&part)) {
        added = append_json(content.parts, TextPart{.text = value->text});
      } else if (const auto* pdf = std::get_if<cail::PdfPart>(&part)) {
        added = append_json(
            content.parts,
            ImagePart{.inlineData = InlineData{.mimeType = "application/pdf",
                                               .data = cail::detail::base64_encode(pdf->bytes)}});
      } else {
        const auto& image = std::get<cail::ImagePart>(part);
        added = append_json(
            content.parts,
            ImagePart{.inlineData = InlineData{.mimeType = image.mime_type,
                                               .data = cail::detail::base64_encode(image.bytes)}});
      }
      if (!added) {
        return std::unexpected(added.error());
      }
    }
    if (message.role != MessageRole::assistant && !message.tool_calls.empty()) {
      return std::unexpected(
          Error{.code = ErrorCode::invalid_tool_call,
                .message = "Only Gemini model messages can contain function calls."});
    }
    for (const auto& call : message.tool_calls) {
      if (call.id.empty() || call.name.empty() || glz::validate_json(call.arguments)) {
        return std::unexpected(
            Error{.code = ErrorCode::invalid_tool_call,
                  .message = "Gemini function calls require an ID, name, and JSON arguments."});
      }
      std::optional<std::string> signature;
      if (call.provider_options.contains("thought_signature")) {
        const auto* value = call.provider_options.at("thought_signature").get_if<std::string>();
        if (!value) {
          return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                       .message = "Gemini thought_signature must be a string."});
        }
        signature = *value;
      }
      auto added =
          append_json(content.parts,
                      CallPart{.functionCall = FunctionCall{.name = call.name,
                                                            .args = glz::raw_json{call.arguments},
                                                            .id = call.id},
                               .thoughtSignature = std::move(signature)});
      if (!added) {
        return std::unexpected(added.error());
      }
    }
    if (content.parts.empty()) {
      return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                   .message = "Gemini messages require content."});
    }
    body.contents.push_back(std::move(content));
  }
  if (!request.tools.empty()) {
    body.tools = std::vector<Tool>{Tool{}};
    for (const auto& tool : request.tools) {
      auto schema = to_json(tool.parameters);
      if (!schema) {
        return std::unexpected(schema.error());
      }
      body.tools->front().functionDeclarations.push_back(
          Declaration{.name = tool.name,
                      .description = tool.description,
                      .parametersJsonSchema = glz::raw_json{std::move(*schema)}});
    }
  }
  if (request.structured_output) {
    auto schema = to_json(request.structured_output->schema);
    if (!schema) {
      return std::unexpected(schema.error());
    }
    auto config = body.generationConfig.value_or(GenerationConfig{});
    config.responseMimeType = "application/json";
    config.responseJsonSchema = glz::raw_json{std::move(*schema)};
    body.generationConfig = std::move(config);
  }
  return body;
}

void apply_usage(GenerationResponse& result, const Usage& usage) {
  if (!result.usage) {
    result.usage.emplace();
  }
  if (usage.promptTokenCount) {
    result.usage->input_tokens = *usage.promptTokenCount;
  }
  const auto candidate_tokens = usage.candidatesTokenCount.value_or(
      result.usage->output_tokens - result.usage->reasoning_tokens.value_or(0));
  if (usage.cachedContentTokenCount) {
    result.usage->cache_read_tokens = *usage.cachedContentTokenCount;
  }
  if (usage.thoughtsTokenCount) {
    result.usage->reasoning_tokens = *usage.thoughtsTokenCount;
  }
  result.usage->output_tokens = candidate_tokens + result.usage->reasoning_tokens.value_or(0);
}

Result<void> apply_chunk(GenerationResponse& result, const ResponseBody& body,
                         const StreamHandler& on_event) {
  if (body.usageMetadata) {
    apply_usage(result, *body.usageMetadata);
    if (on_event) {
      on_event(StreamEvent{UsageUpdate{.usage = *result.usage}});
    }
  }
  if (body.candidates.empty()) {
    return {};
  }
  const auto& candidate = body.candidates.front();
  if (candidate.finishReason) {
    cail::detail::apply_finish_reason(result, candidate.finishReason);
  }
  if (result.finish_reason == FinishReason::stop && !result.tool_calls.empty()) {
    result.finish_reason = FinishReason::tool_calls;
  }
  if (!candidate.content) {
    return {};
  }
  for (const auto& part : candidate.content->parts) {
    if (part.text) {
      if (part.thought.value_or(false)) {
        result.reasoning += *part.text;
        if (on_event) {
          on_event(StreamEvent{ReasoningDelta{.text = *part.text}});
        }
      } else {
        result.text += *part.text;
        if (on_event) {
          on_event(StreamEvent{TextDelta{.text = *part.text}});
        }
      }
    }
    if (part.functionCall) {
      auto& call = *part.functionCall;
      if (call.name.empty() || call.args.str.empty()) {
        return std::unexpected(Error{.code = ErrorCode::provider_response,
                                     .message = "Gemini returned an incomplete function call."});
      }
      if (!call.id || call.id->empty()) {
        return std::unexpected(Error{.code = ErrorCode::provider_response,
                                     .message = "Gemini returned a function call without an ID."});
      }
      cail::ToolCall mapped{
          .id = *call.id,
          .name = call.name,
          .arguments = call.args.str,
          .provider_options = part.thoughtSignature
                                  ? ProviderOptions{{"thought_signature", *part.thoughtSignature}}
                                  : ProviderOptions{},
      };
      if (on_event) {
        on_event(
            StreamEvent{ToolCallReady{.output_index = result.tool_calls.size(), .call = mapped}});
      }
      result.tool_calls.push_back(std::move(mapped));
    }
  }
  if (result.finish_reason == FinishReason::stop && !result.tool_calls.empty()) {
    result.finish_reason = FinishReason::tool_calls;
  }
  return {};
}

} // namespace cail::detail::gemini

namespace cail::detail::gemini {

struct Client::StreamState {
  StreamHandler handler;
  std::stop_token stop;
  cail::detail::SseParser parser;
  GenerationResponse result;
  std::optional<Error> stream_error;
  bool received = false;
  void handle_event(const cail::detail::ServerSentEvent& event) {
    if (stop.stop_requested() || stream_error || event.data.empty()) {
      return;
    }
    ResponseBody chunk;
    if (const auto error = glz::read<glz::opts{.error_on_unknown_keys = false}>(chunk, event.data);
        error) {
      stream_error = Error{.code = ErrorCode::provider_response,
                           .message = glz::format_error(error, event.data)};
    } else {
      received = true;
      auto applied = apply_chunk(result, chunk, handler);
      if (!applied) {
        stream_error = applied.error();
      }
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
    if (!received) {
      return context(Error{.code = ErrorCode::provider_response,
                           .message = "Gemini returned an empty stream."});
    }
    return result;
  }
};

EmbeddingClient::EmbeddingClient(Config config, std::optional<std::size_t> dimensions,
                                 std::unique_ptr<HttpTransport> transport)
    : config_(std::move(config)), dimensions_(dimensions), transport_(std::move(transport)) {}

Result<EmbeddingBatch> EmbeddingClient::embed_many(const std::vector<std::string>& inputs,
                                                   const std::stop_token& stop) const {
  if (stop.stop_requested()) {
    return std::unexpected(generation_cancelled_error());
  }
  auto http = make_http_request(inputs);
  if (!http) {
    return std::unexpected(http.error());
  }
  auto response = transport_->send(*http, stop);
  if (stop.stop_requested()) {
    return std::unexpected(generation_cancelled_error());
  }
  if (!response) {
    return std::unexpected(response.error());
  }
  return decode_http_response(*response, inputs.size());
}

Result<void> EmbeddingClient::embed_many_async(const std::vector<std::string>& inputs,
                                               EmbeddingModel::BatchCompletion complete,
                                               const std::stop_token& stop) const {
  if (stop.stop_requested()) {
    return std::unexpected(generation_cancelled_error());
  }
  if (!complete) {
    return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                 .message = "Async embeddings require a completion handler."});
  }
  auto http = make_http_request(inputs);
  if (!http) {
    return std::unexpected(http.error());
  }
  transport_->send_async(
      std::move(*http),
      [this, input_count = inputs.size(), complete = std::move(complete),
       stop](Result<HttpResponse> response) {
        if (stop.stop_requested()) {
          complete(std::unexpected(generation_cancelled_error()));
        } else if (!response) {
          complete(std::unexpected(response.error()));
        } else {
          complete(decode_http_response(*response, input_count));
        }
      },
      stop);
  return {};
}

Result<HttpRequest>
EmbeddingClient::make_http_request(const std::vector<std::string>& inputs) const {
  if (config_.api_key.empty() || config_.model.empty() || config_.base_url.empty() || !transport_) {
    return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                 .message = "Gemini embeddings require an API key, model, base "
                                            "URL, and transport."});
  }
  BatchEmbedContentsRequest request;
  for (const auto& input : inputs) {
    request.requests.push_back(EmbedContentRequest{
        .model = "models/" + model_id(),
        .content = EmbedContent{.parts = {EmbedPart{.text = input}}},
        .outputDimensionality = dimensions_,
    });
  }
  auto encoded = to_json(request);
  if (!encoded) {
    return std::unexpected(encoded.error());
  }
  HttpRequest http{.url = config_.base_url + "/models/" + model_id() + ":batchEmbedContents",
                   .headers = config_.headers,
                   .body = std::move(*encoded)};
  if (config_.api_key_header.empty()) {
    return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                 .message = "Gemini requires an API key header name."});
  }
  http.headers.push_back(
      {.name = config_.api_key_header, .value = config_.api_key_prefix + config_.api_key});
  http.headers.push_back({.name = "Content-Type", .value = "application/json"});
  cail::detail::append_session_header(http.headers, config_.request_session_header, config_.model);
  return http;
}

Result<EmbeddingBatch> EmbeddingClient::decode_http_response(const HttpResponse& response,
                                                             std::size_t input_count) const {
  const auto context = [&](Error error) {
    return unexpected_with_http_context<EmbeddingBatch>(std::move(error), response);
  };
  if (is_http_error_status(response.status_code)) {
    return context(http_status_error_from_json_body(response));
  }
  BatchEmbedContentsResponse body;
  if (const auto error = glz::read<glz::opts{.error_on_unknown_keys = false}>(body, response.body);
      error) {
    return context(Error{.code = ErrorCode::provider_response,
                         .message = glz::format_error(error, response.body)});
  }
  if (body.embeddings.size() != input_count) {
    return context(Error{.code = ErrorCode::provider_response,
                         .message = "Gemini returned an incomplete embedding batch."});
  }
  EmbeddingBatch result{.embeddings = std::vector<Embedding>(input_count),
                        .model = config_.model,
                        .input_tokens = body.usageMetadata ? body.usageMetadata->promptTokenCount
                                                           : std::nullopt};
  for (std::size_t index = 0; index < body.embeddings.size(); ++index) {
    auto& values = body.embeddings[index].values;
    if (values.empty() || (result.dimensions && values.size() != result.dimensions) ||
        (dimensions_ && values.size() != *dimensions_)) {
      return context(Error{.code = ErrorCode::provider_response,
                           .message = "Gemini returned an embedding with an invalid dimension."});
    }
    result.dimensions = values.size();
    result.embeddings[index] = Embedding{
        .values = std::move(values), .model = config_.model, .dimensions = result.dimensions};
  }
  return result;
}

// The URL template uses "models/{model}", so accept an ID with or without the prefix.
std::string EmbeddingClient::model_id() const {
  return config_.model.starts_with("models/") ? config_.model.substr(7) : config_.model;
}

Client::Client(Config config, std::unique_ptr<HttpTransport> transport)
    : config_(std::move(config)), transport_(std::move(transport)) {}

Result<GenerationResponse> Client::generate(const GenerationRequest& request,
                                            const std::stop_token& stop) const {
  return run(request, {}, stop);
}

Result<GenerationResponse> Client::stream(const GenerationRequest& request,
                                          const StreamHandler& handler,
                                          const std::stop_token& stop) const {
  return run(request, handler, stop);
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
  state->handler = std::move(on_event);
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
  GenerationResponse result;
  ResponseBody parsed;
  if (const auto error =
          glz::read<glz::opts{.error_on_unknown_keys = false}>(parsed, response.body);
      error) {
    return context(Error{.code = ErrorCode::provider_response,
                         .message = glz::format_error(error, response.body)});
  }
  if (parsed.candidates.empty()) {
    return context(
        Error{.code = ErrorCode::provider_response, .message = "Gemini returned no candidates."});
  }
  auto applied = apply_chunk(result, parsed, {});
  return applied ? Result<GenerationResponse>{std::move(result)} : context(applied.error());
}

Result<HttpRequest> Client::make_http_request(const GenerationRequest& request,
                                              bool streaming) const {
  if (config_.api_key.empty() || config_.model.empty() || config_.base_url.empty() || !transport_) {
    return std::unexpected(
        Error{.code = ErrorCode::invalid_configuration,
              .message = "Gemini requires an API key, model, base URL, and transport."});
  }
  auto body = encode(request);
  if (!body) {
    return std::unexpected(body.error());
  }
  auto json = to_json(*body);
  if (!json) {
    return std::unexpected(json.error());
  }
  if (!request.provider_options.empty()) {
    glz::generic options = request.provider_options;
    if (options.is_object() && options.contains("reasoning_effort") &&
        options["reasoning_effort"].is_string()) {
      auto effort = options["reasoning_effort"].get<std::string>();
      options.get<glz::generic::object_t>().erase("reasoning_effort");
      if (!options.contains("generationConfig")) {
        options["generationConfig"] = glz::generic::object_t{};
      }
      auto& config = options["generationConfig"];
      config["thinkingConfig"] = glz::generic::object_t{};
      if (config_.model.starts_with("gemini-2.5")) {
        const int budget = effort == "none"                         ? 0
                           : effort == "low" || effort == "minimal" ? 1024
                           : effort == "medium"                     ? 8192
                           : effort == "high"                       ? 24576
                                                                    : -1;
        config["thinkingConfig"]["thinkingBudget"] = budget;
      } else {
        for (auto& c : effort) {
          if (c >= 'a' && c <= 'z') {
            c = static_cast<char>(c - 'a' + 'A');
          }
        }
        config["thinkingConfig"]["thinkingLevel"] = effort;
      }
    }
    if (body->generationConfig && options.is_object() && options.contains("generationConfig")) {
      auto base = to_json(*body->generationConfig);
      if (!base) {
        return std::unexpected(base.error());
      }
      const auto* override_config = options["generationConfig"].get_if<glz::generic::object_t>();
      if (!override_config) {
        return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                     .message = "generationConfig must be a JSON object."});
      }
      auto merged_config = merge_json_objects(*base, *override_config);
      if (!merged_config) {
        return std::unexpected(merged_config.error());
      }
      if (const auto error = glz::read_json(options["generationConfig"], *merged_config); error) {
        return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                     .message = glz::format_error(error, *merged_config)});
      }
    }
    auto merged = merge_json_objects(*json, options.get<glz::generic::object_t>());
    if (!merged) {
      return std::unexpected(merged.error());
    }
    json = std::move(*merged);
  }
  HttpRequest http{.url = config_.base_url + "/models/" + config_.model +
                          (streaming ? ":streamGenerateContent?alt=sse" : ":generateContent"),
                   .headers = config_.headers,
                   .body = std::move(*json)};
  if (config_.api_key_header.empty()) {
    return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                 .message = "Gemini requires an API key header name."});
  }
  http.headers.push_back(
      {.name = config_.api_key_header, .value = config_.api_key_prefix + config_.api_key});
  http.headers.push_back({.name = "Content-Type", .value = "application/json"});
  http.headers.push_back({.name = "Connection", .value = "close"});
  if (streaming) {
    http.headers.push_back({.name = "Accept", .value = "text/event-stream"});
  }
  cail::detail::append_session_header(http.headers, config_.request_session_header,
                                      request.session_id);
  if (auto middleware = cail::detail::run_before_request(request, http); !middleware) {
    return std::unexpected(middleware.error());
  }
  return http;
}

Result<GenerationResponse> Client::run(const GenerationRequest& request,
                                       const StreamHandler& handler,
                                       const std::stop_token& stop) const {
  if (stop.stop_requested()) {
    return std::unexpected(generation_cancelled_error());
  }
  auto prepared = make_http_request(request, static_cast<bool>(handler));
  if (!prepared) {
    return std::unexpected(prepared.error());
  }
  auto http = std::move(*prepared);
  StreamState state;
  state.handler = handler;
  state.stop = stop;
  auto response = handler ? transport_->stream(
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
  return handler ? state.finish(*response) : decode_http_response(*response);
}

} // namespace cail::detail::gemini

namespace cail {

LanguageModel GeminiProvider::operator()(std::string model_id) const {
  return (*this)(std::move(model_id), make_default_http_transport());
}

LanguageModel GeminiProvider::operator()(std::string model_id,
                                         std::unique_ptr<HttpTransport> transport) const {
  auto key = detail::env_or(settings_.api_key, "GEMINI_API_KEY");
  auto client = std::make_shared<detail::gemini::Client>(
      detail::gemini::Config{.api_key = std::move(key),
                             .model = std::move(model_id),
                             .base_url = settings_.base_url,
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

EmbeddingModel GeminiProvider::embedding_model(std::string model_id,
                                               std::optional<std::size_t> dimensions) const {
  auto client = std::make_shared<detail::gemini::EmbeddingClient>(
      detail::gemini::Config{
          .api_key = detail::env_or(settings_.api_key, "GEMINI_API_KEY"),
          .model = std::move(model_id),
          .base_url = settings_.base_url,
          .headers = settings_.headers,
      },
      dimensions);
  return EmbeddingModel{
      [client](const std::vector<std::string>& inputs, const std::stop_token& stop) {
        return client->embed_many(inputs, stop);
      },
      [client](const std::vector<std::string>& inputs, EmbeddingModel::BatchCompletion complete,
               const std::stop_token& stop) {
        return client->embed_many_async(
            inputs,
            [client, complete = std::move(complete)](Result<EmbeddingBatch> result) {
              complete(std::move(result));
            },
            stop);
      }};
}

} // namespace cail
