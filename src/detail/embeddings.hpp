#pragma once

#include <cail/detail/openai_embeddings.hpp>

#include <cstddef>
#include <optional>
#include <stop_token>
#include <string>
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

class EmbeddingClient {
public:
  explicit EmbeddingClient(EmbeddingClientSettings settings = {});

  [[nodiscard]] Result<EmbeddingBatch> embed_many(const std::vector<std::string>& inputs,
                                                  const std::stop_token& stop = {}) const;

  [[nodiscard]] Result<void> embed_many_async(const std::vector<std::string>& inputs,
                                              EmbeddingModel::BatchCompletion complete,
                                              const std::stop_token& stop = {}) const;

private:
  [[nodiscard]] Result<HttpRequest> make_http_request(const std::vector<std::string>& inputs) const;

  [[nodiscard]] Result<EmbeddingBatch> decode_http_response(const HttpResponse& response,
                                                            std::size_t input_count) const;

  EmbeddingClientSettings settings_;
};

} // namespace cail::detail
