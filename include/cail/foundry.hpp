#pragma once

#include <cail/embedding_model.hpp>
#include <cail/http.hpp>
#include <cail/language_model.hpp>

#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace cail {

struct FoundryTag {
  static constexpr const char* env_var = "AZURE_FOUNDRY_API_KEY";
};

struct FoundryDeployment {
  // Full Responses endpoint URL for this deployment, for example
  // "https://<resource>.cognitiveservices.azure.com/openai/responses?api-version=2025-04-01-preview".
  std::string endpoint;
  std::string deployment;
};

struct FoundrySettings {
  std::string api_key;
};

struct FoundryModelSettings {
  std::string api_key;
  std::string endpoint;
  std::string deployment;
};

class FoundryProvider {
public:
  explicit FoundryProvider(FoundrySettings settings = {}) : settings_(std::move(settings)) {}

  [[nodiscard]] LanguageModel operator()(FoundryDeployment deployment) const;

  [[nodiscard]] LanguageModel operator()(FoundryDeployment deployment,
                                         std::unique_ptr<HttpTransport> transport) const;

  // Azure serves embeddings from the deployment's own URL, which includes the
  // API version, so pass the full embeddings endpoint.
  [[nodiscard]] EmbeddingModel
  embedding_model(std::string model_id, std::string endpoint,
                  std::optional<std::size_t> dimensions = std::nullopt) const;

private:
  FoundrySettings settings_;
};

[[nodiscard]] inline FoundryProvider create_foundry(FoundrySettings settings = {}) {
  return FoundryProvider{std::move(settings)};
}

[[nodiscard]] inline LanguageModel create_foundry_model(FoundryModelSettings settings) {
  return create_foundry({.api_key = std::move(settings.api_key)})(FoundryDeployment{
      .endpoint = std::move(settings.endpoint),
      .deployment = std::move(settings.deployment),
  });
}

inline FoundryProvider foundry{};

} // namespace cail
