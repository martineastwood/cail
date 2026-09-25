#pragma once

#include <cail/error.hpp>
#include <cail/generation.hpp>

#include <functional>
#include <utility>

namespace cail {

class LanguageModel {
    public:
    using GenerateFunction = std::function<Result<GenerationResponse>(const GenerationRequest&)>;
    using StreamFunction = std::function<Result<GenerationResponse>(const GenerationRequest&, const TextDeltaHandler&)>;

    LanguageModel() = default;
    explicit LanguageModel(GenerateFunction generate) : generate_(std::move(generate)) {}
    LanguageModel(GenerateFunction generate, StreamFunction stream)
        : generate_(std::move(generate)), stream_(std::move(stream)) {}

    [[nodiscard]] explicit operator bool() const noexcept { return static_cast<bool>(generate_); }

    [[nodiscard]] Result<GenerationResponse> generate(const GenerationRequest& request) const
    {
        if (!generate_) {
            return std::unexpected(Error{
                .code = ErrorCode::invalid_configuration,
                .message = "The language model has no provider implementation.",
            });
        }
        return generate_(request);
    }

    [[nodiscard]] Result<GenerationResponse> stream(const GenerationRequest& request, const TextDeltaHandler& on_text_delta) const
    {
        if (!stream_) {
            return std::unexpected(Error{
                .code = ErrorCode::invalid_configuration,
                .message = "The language model provider does not support streaming.",
            });
        }
        if (!on_text_delta) {
            return std::unexpected(Error{
                .code = ErrorCode::invalid_configuration,
                .message = "Streaming requires a text delta handler.",
            });
        }
        return stream_(request, on_text_delta);
    }

    [[nodiscard]] Result<GenerationResponse> stream(std::string_view prompt, const TextDeltaHandler& on_text_delta) const
    {
        return stream(GenerationRequest{
            .messages = {Message{.role = MessageRole::user, .content = std::string{prompt}}},
        }, on_text_delta);
    }

    private:
    GenerateFunction generate_;
    StreamFunction stream_;
};

} // namespace cail
