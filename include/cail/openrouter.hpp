#pragma once

#include <cail/detail/chat_completions_preset.hpp>

namespace cail {

struct OpenRouterTag {
    static constexpr const char* endpoint = "https://openrouter.ai/api/v1/chat/completions";
    static constexpr const char* env_var = "OPENROUTER_API_KEY";
};

using OpenRouterSettings = ChatCompletionsPresetSettings<OpenRouterTag>;
using OpenRouterProvider = detail::ChatCompletionsPresetProvider<OpenRouterTag>;

[[nodiscard]] inline OpenRouterProvider create_openrouter(OpenRouterSettings settings = {})
{
    settings.prompt_cache_key = true;
    settings.session_body = true;
    return OpenRouterProvider{std::move(settings)};
}

inline OpenRouterProvider openrouter{};

} // namespace cail
