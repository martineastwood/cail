#pragma once

#include <cail/detail/glaze_http_transport.hpp>
#include <cail/detail/http_context.hpp>
#include <cail/embedding_model.hpp>
#include <cail/json.hpp>

#include <glaze/glaze.hpp>

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace cail::detail::openai {

struct EmbeddingRequestBody {
    std::string model;
    std::vector<std::string> input;
    std::string encoding_format{"float"};
    std::optional<std::size_t> dimensions;
};

struct EmbeddingData {
    std::size_t index{};
    std::vector<float> embedding;
};

struct EmbeddingUsage {
    std::size_t prompt_tokens{};
};

struct EmbeddingResponseBody {
    std::string model;
    std::vector<EmbeddingData> data;
    std::optional<EmbeddingUsage> usage;
};

struct EmbeddingErrorBody {
    struct ProviderError {
        std::string message;
        std::optional<std::string> code;
        std::optional<std::string> type;
    };
    std::optional<ProviderError> error;
};

class EmbeddingClient {
    public:
    EmbeddingClient(std::string api_key, std::string model, std::string base_url,
                    std::optional<std::size_t> dimensions = std::nullopt,
                    std::unique_ptr<HttpTransport> transport = std::make_unique<cail::detail::GlazeHttpTransport>())
        : api_key_(std::move(api_key)), model_(std::move(model)), base_url_(std::move(base_url)),
          dimensions_(dimensions), transport_(std::move(transport)) {}

    [[nodiscard]] Result<EmbeddingBatch> embed_many(const std::vector<std::string>& inputs) const
    {
        if (api_key_.empty() || model_.empty() || base_url_.empty() || !transport_ ||
            (dimensions_ && *dimensions_ == 0)) {
            return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                         .message = "OpenAI embeddings require an API key, model, base URL, transport, and positive dimensions."});
        }
        auto encoded = to_json(EmbeddingRequestBody{.model = model_, .input = inputs, .dimensions = dimensions_});
        if (!encoded) return std::unexpected(encoded.error());
        auto response = transport_->send(HttpRequest{
            .url = base_url_ + "/embeddings",
            .headers = {{.name = "Authorization", .value = "Bearer " + api_key_},
                        {.name = "Content-Type", .value = "application/json"}},
            .body = std::move(*encoded),
        });
        if (!response) return std::unexpected(response.error());
        const auto context = [&](Error error) {
            return unexpected_with_http_context<EmbeddingBatch>(std::move(error), *response);
        };
        if (response->status_code < 200 || response->status_code >= 300) {
            EmbeddingErrorBody body;
            const auto parsed = glz::read<glz::opts{.error_on_unknown_keys = false}>(body, response->body);
            return context(Error{
                .code = ErrorCode::http_status,
                .message = !parsed && body.error ? body.error->message : response->body,
                .provider_code = !parsed && body.error ? body.error->code.value_or("") : "",
                .provider_type = !parsed && body.error ? body.error->type.value_or("") : "",
            });
        }
        EmbeddingResponseBody body;
        if (const auto error = glz::read<glz::opts{.error_on_unknown_keys = false}>(body, response->body); error) {
            return context(Error{.code = ErrorCode::provider_response,
                                 .message = glz::format_error(error, response->body)});
        }
        if (body.model.empty() || body.data.size() != inputs.size()) {
            return context(Error{.code = ErrorCode::provider_response,
                                 .message = "OpenAI returned an incomplete embedding batch."});
        }
        EmbeddingBatch result{.embeddings = std::vector<Embedding>(inputs.size()),
                              .model = body.model,
                              .input_tokens = body.usage ? std::optional{body.usage->prompt_tokens} : std::nullopt};
        std::vector<bool> seen(inputs.size());
        for (auto& item : body.data) {
            if (item.index >= inputs.size() || seen[item.index] || item.embedding.empty() ||
                (result.dimensions && item.embedding.size() != result.dimensions) ||
                (dimensions_ && item.embedding.size() != *dimensions_)) {
                return context(Error{.code = ErrorCode::provider_response,
                                     .message = "OpenAI returned an invalid embedding index or vector dimension."});
            }
            seen[item.index] = true;
            result.dimensions = item.embedding.size();
            result.embeddings[item.index] = Embedding{
                .values = std::move(item.embedding), .model = body.model, .dimensions = result.dimensions};
        }
        return result;
    }

    private:
    std::string api_key_;
    std::string model_;
    std::string base_url_;
    std::optional<std::size_t> dimensions_;
    std::unique_ptr<HttpTransport> transport_;
};

} // namespace cail::detail::openai
