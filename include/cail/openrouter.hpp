#pragma once

#include <cail/chat_completions.hpp>

#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace cail {

struct OpenRouterSettings {
    std::string api_key;
    std::vector<HttpHeader> headers;
};

class OpenRouterProvider {
    public:
    explicit OpenRouterProvider(OpenRouterSettings settings = {}) : settings_(std::move(settings)) {}

    [[nodiscard]] LanguageModel operator()(std::string model_id) const
    {
        auto key = settings_.api_key;
        if (key.empty()) {
            if (const char* env = std::getenv("OPENROUTER_API_KEY"); env && *env) {
                key = env;
            }
        }
        return create_chat_completions({
            .endpoint = "https://openrouter.ai/api/v1/chat/completions",
            .api_key = std::move(key),
            .headers = settings_.headers,
        })(std::move(model_id));
    }

    private:
    OpenRouterSettings settings_;
};

[[nodiscard]] inline OpenRouterProvider create_openrouter(OpenRouterSettings settings = {})
{
    return OpenRouterProvider{std::move(settings)};
}

inline OpenRouterProvider openrouter{};

} // namespace cail
