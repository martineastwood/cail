#pragma once

#include <cail/chat_completions.hpp>
#include <cail/detail/env.hpp>

#include <string>
#include <utility>
#include <vector>

namespace cail {

template <typename Tag>
struct ChatCompletionsPresetSettings {
    std::string api_key;
    std::vector<HttpHeader> headers;
};

namespace detail {

template <typename Tag>
class ChatCompletionsPresetProvider {
    public:
    explicit ChatCompletionsPresetProvider(ChatCompletionsPresetSettings<Tag> settings = {})
        : settings_(std::move(settings)) {}

    [[nodiscard]] LanguageModel operator()(std::string model_id) const
    {
        return create_chat_completions({
            .endpoint = Tag::endpoint,
            .api_key = env_or(settings_.api_key, Tag::env_var),
            .headers = settings_.headers,
        })(std::move(model_id));
    }

    private:
    ChatCompletionsPresetSettings<Tag> settings_;
};

} // namespace detail

} // namespace cail
