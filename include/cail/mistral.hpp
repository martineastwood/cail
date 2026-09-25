#pragma once

#include <cail/chat_completions.hpp>
#include <cail/detail/env.hpp>

#include <string>
#include <utility>
#include <vector>

namespace cail {

struct MistralSettings {
    std::string api_key;
    std::vector<HttpHeader> headers;
};

class MistralProvider {
    public:
    explicit MistralProvider(MistralSettings settings = {}) : settings_(std::move(settings)) {}

    [[nodiscard]] LanguageModel operator()(std::string model_id) const
    {
        auto key = detail::env_or(settings_.api_key, "MISTRAL_API_KEY");
        return create_chat_completions({
            .endpoint = "https://api.mistral.ai/v1/chat/completions",
            .api_key = std::move(key),
            .headers = settings_.headers,
        })(std::move(model_id));
    }

    private:
    MistralSettings settings_;
};

[[nodiscard]] inline MistralProvider create_mistral(MistralSettings settings = {})
{
    return MistralProvider{std::move(settings)};
}

inline MistralProvider mistral{};

} // namespace cail
