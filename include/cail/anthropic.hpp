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

struct AnthropicSettings {
  std::string api_key;
  std::string base_url{"https://api.anthropic.com/v1"};
  std::size_t max_tokens{1024};
  std::vector<HttpHeader> headers;
};

class AnthropicProvider {
public:
  explicit AnthropicProvider(AnthropicSettings settings = {}) : settings_(std::move(settings)) {}

  [[nodiscard]] LanguageModel operator()(std::string model_id) const;

  [[nodiscard]] LanguageModel operator()(std::string model_id,
                                         std::unique_ptr<HttpTransport> transport) const;

  [[nodiscard]] EmbeddingModel
  embedding_model(std::string model_id, std::optional<std::size_t> dimensions = std::nullopt) const;

private:
  AnthropicSettings settings_;
};

[[nodiscard]] inline AnthropicProvider create_anthropic(AnthropicSettings settings = {}) {
  return AnthropicProvider{std::move(settings)};
}

inline AnthropicProvider anthropic{};

} // namespace cail
