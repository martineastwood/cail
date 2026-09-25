#pragma once

#include <cail/detail/env.hpp>
#include <cail/detail/glaze_http_transport.hpp>
#include <cail/language_model.hpp>
#include <cail/openai.hpp>

#include <memory>
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

    [[nodiscard]] LanguageModel operator()(FoundryDeployment deployment) const
    {
        return (*this)(std::move(deployment), std::make_unique<detail::GlazeHttpTransport>());
    }

    [[nodiscard]] LanguageModel operator()(FoundryDeployment deployment,
                                         std::unique_ptr<HttpTransport> transport) const
    {
        auto client = detail::make_openai_responses_client(detail::openai::Config{
            .api_key = detail::env_or(settings_.api_key, FoundryTag::env_var),
            .model = std::move(deployment.deployment),
            .base_url = std::move(deployment.endpoint),
        }, std::move(transport));
        return detail::language_model_from(client);
    }

    private:
    FoundrySettings settings_;
};

[[nodiscard]] inline FoundryProvider create_foundry(FoundrySettings settings = {})
{
    return FoundryProvider{std::move(settings)};
}

[[nodiscard]] inline LanguageModel create_foundry_model(FoundryModelSettings settings)
{
    return create_foundry({.api_key = std::move(settings.api_key)})(FoundryDeployment{
        .endpoint = std::move(settings.endpoint),
        .deployment = std::move(settings.deployment),
    });
}

inline FoundryProvider foundry{};

} // namespace cail
