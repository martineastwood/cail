#pragma once

#include <cail/embedding_model.hpp>
#include <cail/http.hpp>
#include <cail/language_model.hpp>

#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace cail {

enum class OpenCodeService {
  zen,
  go,
};

enum class OpenCodeApiFamily {
  chat_completions,
  responses,
  anthropic_messages,
  gemini,
};

struct OpenCodeSettings {
  std::string api_key;
  OpenCodeService service{OpenCodeService::zen};
  std::string base_url;
  std::size_t anthropic_max_tokens{1024};
};

class OpenCodeProvider {
public:
  explicit OpenCodeProvider(OpenCodeSettings settings) : settings_(std::move(settings)) {}

  [[nodiscard]] LanguageModel operator()(std::string model_id, OpenCodeApiFamily api_family) const;

  [[nodiscard]] LanguageModel operator()(std::string model_id, OpenCodeApiFamily api_family,
                                         std::unique_ptr<HttpTransport> transport) const;

  [[nodiscard]] EmbeddingModel
  embedding_model(std::string model_id, std::optional<std::size_t> dimensions = std::nullopt) const;

private:
  OpenCodeSettings settings_;
};

[[nodiscard]] inline OpenCodeProvider create_opencode(OpenCodeSettings settings) {
  return OpenCodeProvider{std::move(settings)};
}

} // namespace cail
