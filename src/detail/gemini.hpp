#pragma once

#include <cail/gemini.hpp>
#include <glaze/core/common.hpp>

#include <cstddef>
#include <memory>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

namespace cail::detail::gemini {

struct Config {
  std::string api_key;
  std::string model;
  std::string base_url{"https://generativelanguage.googleapis.com/v1beta"};
  std::vector<HttpHeader> headers;
  std::string request_session_header;
  std::string api_key_header{"x-goog-api-key"};
  std::string api_key_prefix;
};
struct TextPart {
  std::string text;
};
struct InlineData {
  std::string mimeType;
  std::string data;
};
struct ImagePart {
  InlineData inlineData;
};
struct FunctionCall {
  std::string name;
  glz::raw_json args;
  std::optional<std::string> id;
};
struct CallPart {
  FunctionCall functionCall;
  std::optional<std::string> thoughtSignature;
};
struct FunctionResponse {
  std::string name;
  glz::raw_json response;
  std::optional<std::string> id;
};
struct ResultPart {
  FunctionResponse functionResponse;
};
struct Content {
  std::string role;
  std::vector<glz::raw_json> parts;
};
struct Declaration {
  std::string name;
  std::string description;
  glz::raw_json parametersJsonSchema;
};
struct Tool {
  std::vector<Declaration> functionDeclarations;
};
struct GenerationConfig {
  std::optional<std::string> responseMimeType;
  std::optional<glz::raw_json> responseJsonSchema;
  std::optional<std::size_t> maxOutputTokens;
  std::optional<double> temperature;
  std::optional<double> topP;
  std::optional<std::vector<std::string>> stopSequences;
};
struct RequestBody {
  std::vector<Content> contents;
  std::optional<Content> systemInstruction;
  std::optional<std::vector<Tool>> tools;
  std::optional<GenerationConfig> generationConfig;
  struct ToolConfig {
    struct FunctionCallingConfig {
      std::string mode;
      std::optional<std::vector<std::string>> allowedFunctionNames;
    } functionCallingConfig;
  };
  std::optional<ToolConfig> toolConfig;
};
struct OutputPart {
  std::optional<std::string> text;
  std::optional<bool> thought;
  std::optional<std::string> thoughtSignature;
  std::optional<FunctionCall> functionCall;
};
struct OutputContent {
  std::vector<OutputPart> parts;
};
struct Candidate {
  std::optional<OutputContent> content;
  std::optional<std::string> finishReason;
};
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
[[nodiscard]] const cail::ToolCall* find_tool_call(const GenerationRequest& request,
                                                   std::string_view id);

[[nodiscard]] Result<RequestBody> encode(const GenerationRequest& request);

void apply_usage(GenerationResponse& result, const Usage& usage);

[[nodiscard]] Result<void> apply_chunk(GenerationResponse& result, const ResponseBody& body,
                                       const StreamHandler& on_event);

// Gemini has no OpenAI-style embeddings endpoint. A batch becomes POST
// /models/{model}:batchEmbedContents with one EmbedContentRequest per input.
struct EmbedPart {
  std::string text;
};
struct EmbedContent {
  std::vector<EmbedPart> parts;
};
struct EmbedContentRequest {
  std::string model;
  EmbedContent content;
  std::optional<std::size_t> outputDimensionality;
};
struct BatchEmbedContentsRequest {
  std::vector<EmbedContentRequest> requests;
};
struct ContentEmbedding {
  std::vector<float> values;
};
struct EmbeddingUsage {
  std::optional<std::size_t> promptTokenCount;
};
struct BatchEmbedContentsResponse {
  std::vector<ContentEmbedding> embeddings;
  std::optional<EmbeddingUsage> usageMetadata;
};

class EmbeddingClient {
public:
  explicit EmbeddingClient(
      Config config, std::optional<std::size_t> dimensions = std::nullopt,
      std::unique_ptr<HttpTransport> transport = cail::make_default_http_transport());

  [[nodiscard]] Result<EmbeddingBatch> embed_many(const std::vector<std::string>& inputs,
                                                  std::stop_token stop = {}) const;

  [[nodiscard]] Result<void> embed_many_async(std::vector<std::string> inputs,
                                              EmbeddingModel::BatchCompletion complete,
                                              std::stop_token stop = {}) const;

private:
  [[nodiscard]] Result<HttpRequest> make_http_request(const std::vector<std::string>& inputs) const;

  [[nodiscard]] Result<EmbeddingBatch> decode_http_response(const HttpResponse& response,
                                                            std::size_t input_count) const;

  // The URL template uses "models/{model}", so accept an ID with or without the prefix.
  [[nodiscard]] std::string model_id() const;

  Config config_;
  std::optional<std::size_t> dimensions_;
  std::unique_ptr<HttpTransport> transport_;
};

class Client {
public:
  explicit Client(Config config,
                  std::unique_ptr<HttpTransport> transport = cail::make_default_http_transport());

  [[nodiscard]] Result<GenerationResponse> generate(const GenerationRequest& request,
                                                    std::stop_token stop = {}) const;
  [[nodiscard]] Result<GenerationResponse> stream(const GenerationRequest& request,
                                                  const StreamHandler& handler,
                                                  std::stop_token stop = {}) const;

  [[nodiscard]] Result<void> generate_async(GenerationRequest request,
                                            LanguageModel::GenerationCompletion complete,
                                            std::stop_token stop = {}) const;

  [[nodiscard]] Result<void> stream_async(GenerationRequest request, StreamHandler on_event,
                                          LanguageModel::GenerationCompletion complete,
                                          std::stop_token stop = {}) const;

private:
  struct StreamState;

  [[nodiscard]] static Result<GenerationResponse>
  decode_http_response(const HttpResponse& response);

  [[nodiscard]] Result<HttpRequest> make_http_request(const GenerationRequest& request,
                                                      bool streaming) const;

  [[nodiscard]] Result<GenerationResponse>
  run(const GenerationRequest& request, const StreamHandler& handler, std::stop_token stop) const;
  Config config_;
  std::unique_ptr<HttpTransport> transport_;
};

} // namespace cail::detail::gemini
