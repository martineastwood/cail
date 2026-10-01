#pragma once

#include <cail/embedding_model.hpp>
#include <cail/http.hpp>
#include <cail/language_model.hpp>

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace cail {

struct GeminiSettings {
  std::string api_key;
  std::string base_url{"https://generativelanguage.googleapis.com/v1beta"};
  std::vector<HttpHeader> headers;
};

class GeminiProvider {
public:
  explicit GeminiProvider(GeminiSettings settings = {}) : settings_(std::move(settings)) {}
  [[nodiscard]] LanguageModel operator()(std::string model_id) const;

  [[nodiscard]] LanguageModel operator()(std::string model_id,
                                         std::unique_ptr<HttpTransport> transport) const;

  [[nodiscard]] EmbeddingModel
  embedding_model(std::string model_id, std::optional<std::size_t> dimensions = std::nullopt) const;

private:
  GeminiSettings settings_;
};

[[nodiscard]] inline GeminiProvider create_gemini(GeminiSettings settings = {}) {
  return GeminiProvider{std::move(settings)};
}

inline GeminiProvider gemini{};

} // namespace cail
