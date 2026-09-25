#pragma once

#include <cail/chat_completions.hpp>
#include <cail/detail/env.hpp>

#include <string>
#include <utility>
#include <vector>

namespace cail {

struct OllamaCloudSettings {
    std::string api_key;
    std::vector<HttpHeader> headers;
};

class OllamaCloudProvider {
    public:
    explicit OllamaCloudProvider(OllamaCloudSettings settings = {}) : settings_(std::move(settings)) {}

    [[nodiscard]] LanguageModel operator()(std::string model_id) const
    {
        auto key = detail::env_or(settings_.api_key, "OLLAMA_API_KEY");
        return create_chat_completions({
            .endpoint = "https://ollama.com/v1/chat/completions",
            .api_key = std::move(key),
            .headers = settings_.headers,
        })(std::move(model_id));
    }

    private:
    OllamaCloudSettings settings_;
};

[[nodiscard]] inline OllamaCloudProvider create_ollama_cloud(OllamaCloudSettings settings = {})
{
    return OllamaCloudProvider{std::move(settings)};
}

inline OllamaCloudProvider ollama_cloud{};

} // namespace cail
