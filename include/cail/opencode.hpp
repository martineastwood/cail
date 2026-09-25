#pragma once

#include <cail/anthropic.hpp>
#include <cail/chat_completions.hpp>
#include <cail/detail/env.hpp>
#include <cail/detail/glaze_http_transport.hpp>
#include <cail/gemini.hpp>
#include <cail/openai.hpp>

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cail {

enum class OpenCodeService {
    zen,
    go,
};

enum class OpenCodeApiFamily {
    chat_completions,
    responses,
    anthropic_messages,
    gemini,
};

struct OpenCodeSettings {
    std::string api_key;
    OpenCodeService service{OpenCodeService::zen};
};

namespace detail::opencode {

inline constexpr std::string_view session_header{"x-opencode-session"};

[[nodiscard]] inline std::string base_url(OpenCodeService service)
{
    switch (service) {
    case OpenCodeService::zen: return "https://opencode.ai/zen/v1";
    case OpenCodeService::go: return "https://opencode.ai/zen/go/v1";
    }
    std::unreachable();
}

[[nodiscard]] inline Result<void> validate_request(const GenerationRequest& request,
                                                   bool has_api_key)
{
    if (!has_api_key) {
        return std::unexpected(Error{
            .code = ErrorCode::invalid_configuration,
            .message = "OpenCode requires an API key (set OPENCODE_API_KEY or pass api_key).",
        });
    }
    if (request.session_id.empty()) {
        return std::unexpected(Error{
            .code = ErrorCode::invalid_configuration,
            .message = "OpenCode requires a caller-supplied, non-empty session_id.",
        });
    }
    return {};
}

template <typename Client>
[[nodiscard]] LanguageModel language_model_from(std::shared_ptr<Client> client,
                                                bool has_api_key,
                                                AdapterCapabilities capabilities)
{
    return LanguageModel{
        [client, has_api_key](const GenerationRequest& request) {
            if (auto valid = validate_request(request, has_api_key); !valid) {
                return Result<GenerationResponse>{std::unexpected(valid.error())};
            }
            return client->generate(request);
        },
        [client, has_api_key](const GenerationRequest& request,
                              const StreamHandler& handler,
                              std::stop_token stop) {
            if (auto valid = validate_request(request, has_api_key); !valid) {
                return Result<GenerationResponse>{std::unexpected(valid.error())};
            }
            return client->stream(request, handler, stop);
        },
        capabilities,
    };
}

} // namespace detail::opencode

class OpenCodeProvider {
public:
    explicit OpenCodeProvider(OpenCodeSettings settings) : settings_(std::move(settings)) {}

    [[nodiscard]] LanguageModel operator()(std::string model_id,
                                           OpenCodeApiFamily api_family) const
    {
        return (*this)(std::move(model_id), api_family,
                       std::make_unique<detail::GlazeHttpTransport>());
    }

    [[nodiscard]] LanguageModel operator()(std::string model_id,
                                           OpenCodeApiFamily api_family,
                                           std::unique_ptr<HttpTransport> transport) const
    {
        const auto key = detail::env_or(settings_.api_key, "OPENCODE_API_KEY");
        const auto root = detail::opencode::base_url(settings_.service);
        const std::string session_header{detail::opencode::session_header};
        switch (api_family) {
        case OpenCodeApiFamily::chat_completions: {
            auto client = std::make_shared<detail::chat_completions::Client>(
                root + "/chat/completions", std::move(model_id), key, std::vector<HttpHeader>{},
                std::move(transport), session_header);
            return detail::opencode::language_model_from(
                std::move(client), !key.empty(), chat_completions_adapter_capabilities());
        }
        case OpenCodeApiFamily::responses: {
            auto client = std::make_shared<detail::openai::Client>(
                detail::openai::Config{
                    .api_key = key,
                    .model = std::move(model_id),
                    .base_url = root,
                    .request_session_header = session_header,
                },
                std::move(transport));
            return detail::opencode::language_model_from(
                std::move(client), !key.empty(), detail::openai_responses_adapter_capabilities());
        }
        case OpenCodeApiFamily::anthropic_messages: {
            auto client = std::make_shared<detail::anthropic::Client>(
                detail::anthropic::Config{
                    .api_key = key,
                    .model = std::move(model_id),
                    .base_url = root,
                    .request_session_header = session_header,
                },
                std::move(transport));
            return detail::opencode::language_model_from(
                std::move(client), !key.empty(), AdapterCapabilities{
                    .image_input = true,
                    .tools = true,
                    .structured_output = true,
                    .reasoning = true,
                });
        }
        case OpenCodeApiFamily::gemini: {
            auto client = std::make_shared<detail::gemini::Client>(
                detail::gemini::Config{
                    .api_key = key,
                    .model = std::move(model_id),
                    .base_url = root,
                    .request_session_header = session_header,
                    .api_key_header = "Authorization",
                    .api_key_prefix = "Bearer ",
                },
                std::move(transport));
            return detail::opencode::language_model_from(
                std::move(client), !key.empty(), AdapterCapabilities{
                    .image_input = true,
                    .tools = true,
                    .structured_output = true,
                    .reasoning = true,
                });
        }
        }
        std::unreachable();
    }

private:
    OpenCodeSettings settings_;
};

[[nodiscard]] inline OpenCodeProvider create_opencode(OpenCodeSettings settings)
{
    return OpenCodeProvider{std::move(settings)};
}

} // namespace cail
