#pragma once

#include <cail/chat_completions.hpp>
#include <cail/detail/env.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace cail {

template <typename Tag>
struct ChatCompletionsPresetSettings {
    std::string api_key;
    std::vector<HttpHeader> headers;
    std::string endpoint{Tag::endpoint};
    std::string request_session_header;
    bool prompt_cache_key = false;
    bool session_body = false;
    bool retain_reasoning_content = true;
};

namespace detail {

template <typename Tag>
class ChatCompletionsPresetProvider {
    public:
    explicit ChatCompletionsPresetProvider(ChatCompletionsPresetSettings<Tag> settings = {})
        : settings_(std::move(settings)) {}

    [[nodiscard]] LanguageModel operator()(std::string model_id) const
    {
        return (*this)(std::move(model_id), std::make_unique<detail::GlazeHttpTransport>());
    }

    [[nodiscard]] LanguageModel operator()(std::string model_id,
                                           std::unique_ptr<HttpTransport> transport) const
    {
        return create_chat_completions({
            .endpoint = settings_.endpoint,
            .api_key = env_or(settings_.api_key, Tag::env_var),
            .headers = settings_.headers,
            .request_session_header = settings_.request_session_header,
            .prompt_cache_key = settings_.prompt_cache_key,
            .session_body = settings_.session_body,
            .retain_reasoning_content = settings_.retain_reasoning_content,
        })(std::move(model_id), std::move(transport));
    }

    private:
    ChatCompletionsPresetSettings<Tag> settings_;
};

} // namespace detail

} // namespace cail
