#pragma once

#include <cail/detail/chat_completions_preset.hpp>

namespace cail {

struct MistralTag {
    static constexpr const char* endpoint = "https://api.mistral.ai/v1/chat/completions";
    static constexpr const char* env_var = "MISTRAL_API_KEY";
};

using MistralSettings = ChatCompletionsPresetSettings<MistralTag>;
using MistralProvider = detail::ChatCompletionsPresetProvider<MistralTag>;

[[nodiscard]] inline MistralProvider create_mistral(MistralSettings settings = {})
{
    return MistralProvider{std::move(settings)};
}

inline MistralProvider mistral{};

} // namespace cail
