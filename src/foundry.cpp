#include "detail/openai.hpp"
#include <cail/detail/env.hpp>
#include <cail/detail/openai_embeddings.hpp>
#include <cail/foundry.hpp>

namespace cail {

LanguageModel FoundryProvider::operator()(FoundryDeployment deployment) const {
  return (*this)(std::move(deployment), make_default_http_transport());
}

LanguageModel FoundryProvider::operator()(FoundryDeployment deployment,
                                          std::unique_ptr<HttpTransport> transport) const {
  auto client = detail::make_openai_responses_client(
      detail::openai::Config{
          .api_key = detail::env_or(settings_.api_key, FoundryTag::env_var),
          .model = std::move(deployment.deployment),
          .base_url = std::move(deployment.endpoint),
      },
      std::move(transport));
  return detail::language_model_from(client);
}

EmbeddingModel FoundryProvider::embedding_model(std::string model_id, std::string endpoint,
                                                std::optional<std::size_t> dimensions) const {
  return detail::make_embedding_model(detail::EmbeddingClientSettings{
      .endpoint = std::move(endpoint),
      .api_key = detail::env_or(settings_.api_key, FoundryTag::env_var),
      .model = std::move(model_id),
      .dimensions = dimensions,
  });
}

} // namespace cail
