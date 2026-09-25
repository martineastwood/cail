#pragma once

#include <cail/detail/chat_completions_preset.hpp>

namespace cail {

struct OllamaCloudTag {
    static constexpr const char* endpoint = "https://ollama.com/v1/chat/completions";
    static constexpr const char* env_var = "OLLAMA_API_KEY";
};

using OllamaCloudSettings = ChatCompletionsPresetSettings<OllamaCloudTag>;
using OllamaCloudProvider = detail::ChatCompletionsPresetProvider<OllamaCloudTag>;

[[nodiscard]] inline OllamaCloudProvider create_ollama_cloud(OllamaCloudSettings settings = {})
{
    return OllamaCloudProvider{std::move(settings)};
}

inline OllamaCloudProvider ollama_cloud{};

} // namespace cail
