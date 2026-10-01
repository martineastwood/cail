#pragma once

#include <cail/embedding_model.hpp>
#include <cail/http.hpp>
#include <cail/language_model.hpp>

#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace cail {

struct OpenAITag {
  static constexpr const char* env_var = "OPENAI_API_KEY";
  static constexpr const char* default_base_url = "https://api.openai.com/v1";
};

struct OpenAIProviderSettings {
  std::string api_key;
  std::string base_url{OpenAITag::default_base_url};
};

class OpenAIProvider {
public:
  explicit OpenAIProvider(OpenAIProviderSettings settings = {}) : settings_(std::move(settings)) {}

  [[nodiscard]] LanguageModel operator()(std::string model_id) const;

  [[nodiscard]] LanguageModel operator()(std::string model_id,
                                         std::unique_ptr<HttpTransport> transport) const;

  [[nodiscard]] EmbeddingModel
  embedding_model(std::string model_id, std::optional<std::size_t> dimensions = std::nullopt) const;

private:
  OpenAIProviderSettings settings_;
};

[[nodiscard]] inline OpenAIProvider create_openai(OpenAIProviderSettings settings = {}) {
  return OpenAIProvider{std::move(settings)};
}

inline OpenAIProvider openai{};

} // namespace cail
