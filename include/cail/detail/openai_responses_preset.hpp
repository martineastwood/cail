#pragma once

#include <cail/detail/env.hpp>
#include <cail/detail/glaze_http_transport.hpp>
#include <cail/http.hpp>
#include <cail/language_model.hpp>

#include <memory>
#include <string>
#include <utility>

namespace cail::detail::openai {
struct Config;
class Client;
} // namespace cail::detail::openai

namespace cail {

template <typename Tag>
struct OpenAiResponsesPresetSettings {
    std::string api_key;
    std::string base_url{Tag::default_base_url};
};

namespace detail {

[[nodiscard]] constexpr AdapterCapabilities openai_responses_adapter_capabilities();
[[nodiscard]] std::shared_ptr<openai::Client>
make_openai_responses_client(openai::Config config, std::unique_ptr<HttpTransport> transport);
[[nodiscard]] LanguageModel language_model_from(const std::shared_ptr<openai::Client>& client);

template <typename Tag>
class OpenAiResponsesPresetProvider {
    public:
    explicit OpenAiResponsesPresetProvider(OpenAiResponsesPresetSettings<Tag> settings = {})
        : settings_(std::move(settings)) {}

    [[nodiscard]] LanguageModel operator()(std::string model_id) const
    {
        return (*this)(std::move(model_id), std::make_unique<GlazeHttpTransport>());
    }

    [[nodiscard]] LanguageModel operator()(std::string model_id,
                                           std::unique_ptr<HttpTransport> transport) const
    {
        auto client = make_openai_responses_client(openai::Config{
            .api_key = env_or(settings_.api_key, Tag::env_var),
            .model = std::move(model_id),
            .base_url = settings_.base_url,
            .prompt_cache_key = true,
        }, std::move(transport));
        return language_model_from(client);
    }

    private:
    OpenAiResponsesPresetSettings<Tag> settings_;
};

} // namespace detail

} // namespace cail
