#pragma once

#include <cail/error.hpp>

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cail {

struct Embedding {
    std::vector<float> values;
    std::string model;
    std::size_t dimensions{};
};

struct EmbeddingBatch {
    std::vector<Embedding> embeddings;
    std::string model;
    std::size_t dimensions{};
    std::optional<std::size_t> input_tokens;
};

class EmbeddingModel {
    public:
    using EmbedManyFunction = std::function<Result<EmbeddingBatch>(const std::vector<std::string>&)>;

    EmbeddingModel() = default;
    explicit EmbeddingModel(EmbedManyFunction embed_many) : embed_many_(std::move(embed_many)) {}

    [[nodiscard]] explicit operator bool() const noexcept { return static_cast<bool>(embed_many_); }

    [[nodiscard]] Result<EmbeddingBatch> embed_many(const std::vector<std::string>& inputs) const
    {
        if (!embed_many_) {
            return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                         .message = "The embedding model has no provider implementation."});
        }
        if (inputs.empty()) {
            return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                         .message = "Embedding requires at least one input."});
        }
        for (const auto& input : inputs) {
            if (input.empty()) {
                return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                             .message = "Embedding inputs cannot be empty."});
            }
        }
        return embed_many_(inputs);
    }

    [[nodiscard]] Result<Embedding> embed(std::string_view input) const
    {
        auto batch = embed_many({std::string{input}});
        if (!batch) return std::unexpected(batch.error());
        if (batch->embeddings.size() != 1) {
            return std::unexpected(Error{.code = ErrorCode::provider_response,
                                         .message = "The embedding provider returned the wrong number of vectors."});
        }
        return std::move(batch->embeddings.front());
    }

    private:
    EmbedManyFunction embed_many_;
};

} // namespace cail
