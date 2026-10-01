#pragma once

#include <cail/embedding_model.hpp>
#include <cail/http.hpp>

#include <memory>
#include <optional>
#include <string>

namespace cail::detail {

struct EmbeddingClientSettings {
  // Full embeddings URL, for example "https://api.openai.com/v1/embeddings".
  std::string endpoint;
  std::string api_key;
  std::string model;
  std::optional<std::size_t> dimensions;
  std::unique_ptr<HttpTransport> transport = make_default_http_transport();
};

[[nodiscard]] EmbeddingModel make_embedding_model(EmbeddingClientSettings settings);

[[nodiscard]] EmbeddingModel
make_openai_style_embedding_model(std::string endpoint, std::string api_key, std::string model,
                                  std::optional<std::size_t> dimensions = std::nullopt);

} // namespace cail::detail
