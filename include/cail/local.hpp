#pragma once

#include <cail/detail/chat_completions_preset.hpp>

namespace cail {

struct LocalTag {
    static constexpr const char* endpoint = "http://127.0.0.1:8080/v1/chat/completions";
    // Local servers usually need no key, so there is no environment variable.
    static constexpr const char* env_var = nullptr;
};

using LocalSettings = ChatCompletionsPresetSettings<LocalTag>;
using LocalProvider = detail::ChatCompletionsPresetProvider<LocalTag>;

[[nodiscard]] inline LocalProvider create_local(LocalSettings settings = {})
{
    return LocalProvider{std::move(settings)};
}

inline LocalProvider local{};

} // namespace cail