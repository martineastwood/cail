#pragma once

#include <cail/detail/chat_completions_preset.hpp>

namespace cail {

struct HyperTag {
    static constexpr const char* endpoint = "https://hyper.charm.land/v1/chat/completions";
    static constexpr const char* env_var = "HYPER_API_KEY";
};

using HyperSettings = ChatCompletionsPresetSettings<HyperTag>;
using HyperProvider = detail::ChatCompletionsPresetProvider<HyperTag>;

[[nodiscard]] inline HyperProvider create_hyper(HyperSettings settings = {})
{
    return HyperProvider{std::move(settings)};
}

inline HyperProvider hyper{};

} // namespace cail
