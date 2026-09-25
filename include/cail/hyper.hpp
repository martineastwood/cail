#pragma once

#include <cail/chat_completions.hpp>
#include <cail/detail/env.hpp>

#include <string>
#include <utility>
#include <vector>

namespace cail {

struct HyperSettings {
    std::string api_key;
    std::vector<HttpHeader> headers;
};

class HyperProvider {
    public:
    explicit HyperProvider(HyperSettings settings = {}) : settings_(std::move(settings)) {}

    [[nodiscard]] LanguageModel operator()(std::string model_id) const
    {
        auto key = detail::env_or(settings_.api_key, "HYPER_API_KEY");
        return create_chat_completions({
            .endpoint = "https://hyper.charm.land/v1/chat/completions",
            .api_key = std::move(key),
            .headers = settings_.headers,
        })(std::move(model_id));
    }

    private:
    HyperSettings settings_;
};

[[nodiscard]] inline HyperProvider create_hyper(HyperSettings settings = {})
{
    return HyperProvider{std::move(settings)};
}

inline HyperProvider hyper{};

} // namespace cail
