#pragma once

#include <cail/detail/glaze_http_transport.hpp>
#include <cail/detail/http_context.hpp>
#include <cail/embedding_model.hpp>
#include <cail/json.hpp>

#include <glaze/glaze.hpp>

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace cail::detail {

// The OpenAI embeddings wire format. OpenAI, Mistral, OpenRouter, Ollama Cloud,
// and most self-hosted servers accept `{"model", "input"}` at a dedicated URL,
// so one client serves all of them.
struct EmbeddingRequestBody {
  std::string model;
  std::vector<std::string> input;
  std::string encoding_format{"float"};
  std::optional<std::size_t> dimensions;
};

struct EmbeddingData {
  std::size_t index{};
  std::vector<float> embedding;
};

struct EmbeddingUsage {
  std::size_t prompt_tokens{};
};

struct EmbeddingResponseBody {
  std::string model;
  std::vector<EmbeddingData> data;
  std::optional<EmbeddingUsage> usage;
};

struct EmbeddingClientSettings {
  // Full embeddings URL, for example "https://api.openai.com/v1/embeddings".
  std::string endpoint;
  std::string api_key;
  std::string model;
  std::optional<std::size_t> dimensions;
  std::unique_ptr<HttpTransport> transport = make_default_http_transport();
};

class EmbeddingClient {
public:
  explicit EmbeddingClient(EmbeddingClientSettings settings = {})
      : settings_(std::move(settings)) {}

  [[nodiscard]] Result<EmbeddingBatch> embed_many(const std::vector<std::string>& inputs) const {
    if (settings_.model.empty() || settings_.endpoint.empty() || !settings_.transport ||
        (settings_.dimensions && *settings_.dimensions == 0)) {
      return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                   .message = "Embeddings require a model, endpoint, transport, "
                                              "and positive dimensions."});
    }
    auto encoded = to_json(EmbeddingRequestBody{
        .model = settings_.model, .input = inputs, .dimensions = settings_.dimensions});
    if (!encoded)
      return std::unexpected(encoded.error());
    std::vector<HttpHeader> headers;
    if (!settings_.api_key.empty()) {
      headers.push_back({.name = "Authorization", .value = "Bearer " + settings_.api_key});
    }
    headers.push_back({.name = "Content-Type", .value = "application/json"});
    auto response = settings_.transport->send(HttpRequest{
        .url = settings_.endpoint,
        .headers = std::move(headers),
        .body = std::move(*encoded),
    });
    if (!response)
      return std::unexpected(response.error());
    const auto context = [&](Error error) {
      return unexpected_with_http_context<EmbeddingBatch>(std::move(error), *response);
    };
    if (is_http_error_status(response->status_code)) {
      return context(http_status_error_from_json_body(*response));
    }
    EmbeddingResponseBody body;
    if (const auto error =
            glz::read<glz::opts{.error_on_unknown_keys = false}>(body, response->body);
        error) {
      return context(Error{.code = ErrorCode::provider_response,
                           .message = glz::format_error(error, response->body)});
    }
    if (body.model.empty() || body.data.size() != inputs.size()) {
      return context(Error{.code = ErrorCode::provider_response,
                           .message = "The embeddings endpoint returned an incomplete batch."});
    }
    EmbeddingBatch result{.embeddings = std::vector<Embedding>(inputs.size()),
                          .model = body.model,
                          .input_tokens =
                              body.usage ? std::optional{body.usage->prompt_tokens} : std::nullopt};
    std::vector<bool> seen(inputs.size());
    for (auto& item : body.data) {
      if (item.index >= inputs.size() || seen[item.index] || item.embedding.empty() ||
          (result.dimensions && item.embedding.size() != result.dimensions) ||
          (settings_.dimensions && item.embedding.size() != *settings_.dimensions)) {
        return context(Error{.code = ErrorCode::provider_response,
                             .message =
                                 "The embeddings endpoint returned an invalid index or vector "
                                 "dimension."});
      }
      seen[item.index] = true;
      result.dimensions = item.embedding.size();
      result.embeddings[item.index] = Embedding{.values = std::move(item.embedding),
                                                .model = body.model,
                                                .dimensions = result.dimensions};
    }
    return result;
  }

private:
  EmbeddingClientSettings settings_;
};

[[nodiscard]] inline EmbeddingModel make_embedding_model(EmbeddingClientSettings settings) {
  auto client = std::make_shared<EmbeddingClient>(std::move(settings));
  return EmbeddingModel{
      [client](const std::vector<std::string>& inputs) { return client->embed_many(inputs); }};
}

} // namespace cail::detail
