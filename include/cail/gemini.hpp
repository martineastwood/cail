#pragma once

#include <cail/detail/base64.hpp>
#include <cail/detail/glaze_http_transport.hpp>
#include <cail/detail/sse.hpp>
#include <cail/generation.hpp>
#include <cail/json.hpp>
#include <cail/language_model.hpp>

#include <glaze/glaze.hpp>

#include <cstdlib>
#include <memory>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cail::detail::gemini {

struct Config {
    std::string api_key;
    std::string model;
    std::string base_url{"https://generativelanguage.googleapis.com/v1beta"};
    std::vector<HttpHeader> headers;
};
struct TextPart { std::string text; };
struct InlineData { std::string mimeType; std::string data; };
struct ImagePart { InlineData inlineData; };
struct FunctionCall { std::string name; glz::raw_json args; std::optional<std::string> id; };
struct CallState { std::optional<std::string> id; std::optional<std::string> signature; };
struct CallPart { FunctionCall functionCall; std::optional<std::string> thoughtSignature; };
struct FunctionResponse { std::string name; glz::raw_json response; std::optional<std::string> id; };
struct ResultPart { FunctionResponse functionResponse; };
struct Content { std::string role; std::vector<glz::raw_json> parts; };
struct Declaration { std::string name; std::string description; glz::raw_json parametersJsonSchema; };
struct Tool { std::vector<Declaration> functionDeclarations; };
struct GenerationConfig {
    std::optional<std::string> responseMimeType;
    std::optional<glz::raw_json> responseJsonSchema;
};
struct RequestBody {
    std::vector<Content> contents;
    std::optional<Content> systemInstruction;
    std::optional<std::vector<Tool>> tools;
    std::optional<GenerationConfig> generationConfig;
};
struct OutputPart {
    std::optional<std::string> text;
    std::optional<bool> thought;
    std::optional<std::string> thoughtSignature;
    std::optional<FunctionCall> functionCall;
};
struct OutputContent { std::vector<OutputPart> parts; };
struct Candidate { std::optional<OutputContent> content; std::optional<std::string> finishReason; };
struct Usage {
    std::optional<std::size_t> promptTokenCount;
    std::optional<std::size_t> candidatesTokenCount;
    std::optional<std::size_t> cachedContentTokenCount;
    std::optional<std::size_t> thoughtsTokenCount;
};
struct ResponseBody {
    std::vector<Candidate> candidates;
    std::optional<Usage> usageMetadata;
};
struct ProviderError { std::string message; std::optional<std::string> status; };
struct ErrorBody { std::optional<ProviderError> error; };

template <typename T>
[[nodiscard]] inline Result<void> append(std::vector<glz::raw_json>& parts, const T& part)
{
    auto encoded = to_json(part);
    if (!encoded) return std::unexpected(encoded.error());
    parts.emplace_back(std::move(*encoded));
    return {};
}

[[nodiscard]] inline Result<RequestBody> encode(const GenerationRequest& request)
{
    if (request.messages.empty() || request.continuation_token)
        return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
            .message = "Gemini requires messages and does not accept continuation tokens."});
    RequestBody body;
    for (const auto& message : request.messages) {
        if (message.role == MessageRole::developer)
            return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                .message = "Gemini does not accept developer-role messages."});
        if (message.role == MessageRole::system) {
            if (!body.contents.empty() || body.systemInstruction || !message.tool_calls.empty())
                return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                    .message = "Gemini requires one system message before conversation messages."});
            Content system{.role = "system"};
            for (const auto& part : message.content) {
                const auto* value = std::get_if<cail::TextPart>(&part);
                if (!value) return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                    .message = "Gemini system instructions must contain text."});
                if (auto added = append(system.parts, TextPart{.text = value->text}); !added)
                    return std::unexpected(added.error());
            }
            body.systemInstruction = std::move(system);
            continue;
        }
        if (message.role == MessageRole::tool) {
            if (body.contents.empty() ||
                message.content.size() != 1 || !std::holds_alternative<cail::TextPart>(message.content.front()))
                return std::unexpected(Error{.code = ErrorCode::invalid_tool_call,
                    .message = "Gemini tool results require a preceding model call and one JSON text part."});
            const auto& calls = request.messages;
            std::string name;
            for (auto it = calls.rbegin(); it != calls.rend(); ++it) {
                for (const auto& call : it->tool_calls) if (call.id == message.tool_call_id) name = call.name;
                if (!name.empty()) break;
            }
            const auto& output = std::get<cail::TextPart>(message.content.front()).text;
            if (name.empty() || glz::validate_json(output))
                return std::unexpected(Error{.code = ErrorCode::invalid_tool_call,
                    .message = "Gemini tool result requires a known call ID and JSON output."});
            auto state = from_json<CallState>(message.tool_call_id);
            if (!state) return std::unexpected(state.error());
            auto added = append(body.contents.emplace_back(Content{.role = "user"}).parts,
                ResultPart{.functionResponse = FunctionResponse{
                    .name = name, .response = glz::raw_json{output}, .id = state->id}});
            if (!added) return std::unexpected(added.error());
            continue;
        }
        Content content{.role = message.role == MessageRole::assistant ? "model" : "user"};
        for (const auto& part : message.content) {
            Result<void> added;
            if (const auto* value = std::get_if<cail::TextPart>(&part))
                added = append(content.parts, TextPart{.text = value->text});
            else {
                const auto& image = std::get<cail::ImagePart>(part);
                added = append(content.parts, ImagePart{.inlineData = InlineData{
                    .mimeType = image.mime_type, .data = cail::detail::base64_encode(image.bytes)}});
            }
            if (!added) return std::unexpected(added.error());
        }
        if (message.role != MessageRole::assistant && !message.tool_calls.empty())
            return std::unexpected(Error{.code = ErrorCode::invalid_tool_call,
                .message = "Only Gemini model messages can contain function calls."});
        for (const auto& call : message.tool_calls) {
            if (call.name.empty() || glz::validate_json(call.arguments))
                return std::unexpected(Error{.code = ErrorCode::invalid_tool_call,
                    .message = "Gemini function calls require a name and JSON arguments."});
            auto state = from_json<CallState>(call.id);
            if (!state) return std::unexpected(state.error());
            auto added = append(content.parts, CallPart{
                .functionCall = FunctionCall{
                    .name = call.name, .args = glz::raw_json{call.arguments}, .id = state->id},
                .thoughtSignature = state->signature});
            if (!added) return std::unexpected(added.error());
        }
        if (content.parts.empty()) return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
            .message = "Gemini messages require content."});
        body.contents.push_back(std::move(content));
    }
    if (!request.tools.empty()) {
        body.tools = std::vector<Tool>{Tool{}};
        for (const auto& tool : request.tools) {
            auto schema = to_json(tool.parameters);
            if (!schema) return std::unexpected(schema.error());
            body.tools->front().functionDeclarations.push_back(Declaration{
                .name = tool.name, .description = tool.description,
                .parametersJsonSchema = glz::raw_json{std::move(*schema)}});
        }
    }
    if (request.structured_output) {
        auto schema = to_json(request.structured_output->schema);
        if (!schema) return std::unexpected(schema.error());
        body.generationConfig = GenerationConfig{
            .responseMimeType = "application/json", .responseJsonSchema = glz::raw_json{std::move(*schema)}};
    }
    return body;
}

inline void apply_usage(GenerationResponse& result, const Usage& usage)
{
    if (!result.usage) result.usage.emplace();
    if (usage.promptTokenCount) result.usage->input_tokens = *usage.promptTokenCount;
    if (usage.candidatesTokenCount) result.usage->output_tokens = *usage.candidatesTokenCount;
    if (usage.cachedContentTokenCount) result.usage->cache_read_tokens = *usage.cachedContentTokenCount;
    if (usage.thoughtsTokenCount) result.usage->reasoning_tokens = *usage.thoughtsTokenCount;
}

[[nodiscard]] inline Result<void> apply_chunk(GenerationResponse& result, const ResponseBody& body,
    const StreamHandler& on_event)
{
    if (body.usageMetadata) {
        apply_usage(result, *body.usageMetadata);
        if (on_event) on_event(StreamEvent{UsageUpdate{.usage = *result.usage}});
    }
    if (body.candidates.empty()) return {};
    const auto& candidate = body.candidates.front();
    if (candidate.finishReason) {
        if (*candidate.finishReason == "MAX_TOKENS") result.status = GenerationStatus::incomplete;
        else if (*candidate.finishReason == "SAFETY" || *candidate.finishReason == "RECITATION" ||
                 *candidate.finishReason == "PROHIBITED_CONTENT") result.status = GenerationStatus::refused;
        else if (*candidate.finishReason != "STOP")
            return std::unexpected(Error{.code = ErrorCode::provider_response,
                .message = "Gemini stopped with reason " + *candidate.finishReason + "."});
    }
    if (!candidate.content) return {};
    for (const auto& part : candidate.content->parts) {
        if (part.text) {
            if (part.thought.value_or(false)) {
                result.reasoning += *part.text;
                if (on_event) on_event(StreamEvent{ReasoningDelta{.text = *part.text}});
            } else {
                result.text += *part.text;
                if (on_event) on_event(StreamEvent{TextDelta{.text = *part.text}});
            }
        }
        if (part.functionCall) {
            auto& call = *part.functionCall;
            if (call.name.empty() || call.args.str.empty())
                return std::unexpected(Error{.code = ErrorCode::provider_response,
                    .message = "Gemini returned an incomplete function call."});
            auto state = to_json(CallState{.id = call.id, .signature = part.thoughtSignature});
            if (!state) return std::unexpected(state.error());
            cail::ToolCall mapped{.id = std::move(*state),
                .name = call.name, .arguments = call.args.str};
            if (on_event) on_event(StreamEvent{ToolCallReady{
                .output_index = result.tool_calls.size(), .call = mapped}});
            result.tool_calls.push_back(std::move(mapped));
        }
    }
    return {};
}

class Client {
public:
    explicit Client(Config config, std::unique_ptr<HttpTransport> transport =
        std::make_unique<cail::detail::GlazeHttpTransport>())
        : config_(std::move(config)), transport_(std::move(transport)) {}

    [[nodiscard]] Result<GenerationResponse> generate(const GenerationRequest& request) const
    { return run(request, {}, {}); }
    [[nodiscard]] Result<GenerationResponse> stream(const GenerationRequest& request,
        const StreamHandler& handler, std::stop_token stop = {}) const
    {
        if (!handler) return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
            .message = "Streaming requires an event handler."});
        return run(request, handler, stop);
    }

private:
    [[nodiscard]] Result<GenerationResponse> run(const GenerationRequest& request,
        const StreamHandler& handler, std::stop_token stop) const
    {
        if (stop.stop_requested()) return std::unexpected(Error{.code = ErrorCode::cancelled,
            .message = "Generation was cancelled."});
        if (config_.api_key.empty() || config_.model.empty() || config_.base_url.empty() || !transport_)
            return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                .message = "Gemini requires an API key, model, base URL, and transport."});
        auto body = encode(request);
        if (!body) return std::unexpected(body.error());
        auto json = to_json(*body);
        if (!json) return std::unexpected(json.error());
        HttpRequest http{.url = config_.base_url + "/models/" + config_.model +
            (handler ? ":streamGenerateContent?alt=sse" : ":generateContent"),
            .headers = config_.headers, .body = std::move(*json)};
        http.headers.push_back({.name = "x-goog-api-key", .value = config_.api_key});
        http.headers.push_back({.name = "Content-Type", .value = "application/json"});
        http.headers.push_back({.name = "Connection", .value = "close"});
        if (handler) http.headers.push_back({.name = "Accept", .value = "text/event-stream"});
        cail::detail::SseParser parser;
        GenerationResponse result;
        std::optional<Error> stream_error;
        bool received = false;
        const auto handle_event = [&](const cail::detail::ServerSentEvent& event) {
            if (stream_error || event.data.empty()) return;
            ResponseBody chunk;
            if (const auto error = glz::read<glz::opts{.error_on_unknown_keys = false}>(chunk, event.data); error)
                stream_error = Error{.code = ErrorCode::provider_response,
                    .message = glz::format_error(error, event.data)};
            else {
                received = true;
                auto applied = apply_chunk(result, chunk, handler);
                if (!applied) stream_error = applied.error();
            }
        };
        auto response = handler ? transport_->stream(http, [&](std::string_view bytes) {
            parser.feed(bytes, handle_event);
        }, stop) : transport_->send(http);
        if (!response) return std::unexpected(response.error());
        if (stop.stop_requested()) return std::unexpected(Error{.code = ErrorCode::cancelled,
            .message = "Generation was cancelled."});
        const auto context = [&](Error error) -> Result<GenerationResponse> {
            error.http_status = response->status_code;
            for (const auto& header : response->headers)
                if (header.name == "x-request-id" || header.name == "X-Request-Id") error.request_id = header.value;
            return std::unexpected(std::move(error));
        };
        if (response->status_code < 200 || response->status_code >= 300) {
            ErrorBody error;
            const auto parsed = glz::read<glz::opts{.error_on_unknown_keys = false}>(error, response->body);
            return context(Error{.code = ErrorCode::http_status,
                .message = !parsed && error.error ? error.error->message : response->body,
                .provider_type = !parsed && error.error ? error.error->status.value_or("") : ""});
        }
        if (handler) {
            parser.finish(handle_event);
            if (stream_error) return context(*stream_error);
            if (!received) return context(Error{.code = ErrorCode::provider_response,
                .message = "Gemini returned an empty stream."});
            return result;
        }
        ResponseBody parsed;
        if (const auto error = glz::read<glz::opts{.error_on_unknown_keys = false}>(parsed, response->body); error)
            return context(Error{.code = ErrorCode::provider_response,
                .message = glz::format_error(error, response->body)});
        if (parsed.candidates.empty()) return context(Error{.code = ErrorCode::provider_response,
            .message = "Gemini returned no candidates."});
        auto applied = apply_chunk(result, parsed, {});
        return applied ? Result<GenerationResponse>{std::move(result)} : context(applied.error());
    }
    Config config_;
    std::unique_ptr<HttpTransport> transport_;
};

} // namespace cail::detail::gemini

namespace cail {

struct GeminiSettings {
    std::string api_key;
    std::string base_url{"https://generativelanguage.googleapis.com/v1beta"};
    std::vector<HttpHeader> headers;
};

class GeminiProvider {
public:
    explicit GeminiProvider(GeminiSettings settings = {}) : settings_(std::move(settings)) {}
    [[nodiscard]] LanguageModel operator()(std::string model_id) const
    {
        auto key = settings_.api_key;
        if (key.empty()) if (const char* env = std::getenv("GEMINI_API_KEY"); env && *env) key = env;
        auto client = std::make_shared<detail::gemini::Client>(detail::gemini::Config{
            .api_key = std::move(key), .model = std::move(model_id),
            .base_url = settings_.base_url, .headers = settings_.headers});
        return LanguageModel{
            [client](const GenerationRequest& request) { return client->generate(request); },
            [client](const GenerationRequest& request, const StreamHandler& handler, std::stop_token stop) {
                return client->stream(request, handler, stop);
            }, LanguageModelCapabilities{.image_input = true, .tools = true,
                .structured_output = true, .reasoning = true}};
    }
private:
    GeminiSettings settings_;
};

[[nodiscard]] inline GeminiProvider create_gemini(GeminiSettings settings = {})
{ return GeminiProvider{std::move(settings)}; }

inline GeminiProvider gemini{};

} // namespace cail
