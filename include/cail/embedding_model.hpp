#pragma once

#include <cail/error.hpp>

#include <cstddef>
#include <functional>
#include <optional>
#include <stop_token>
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
  using EmbeddingCompletion = std::function<void(Result<Embedding>)>;
  using BatchCompletion = std::function<void(Result<EmbeddingBatch>)>;
  using EmbedManyFunction =
      std::function<Result<EmbeddingBatch>(const std::vector<std::string>&, std::stop_token)>;
  using EmbedManyAsyncFunction =
      std::function<Result<void>(std::vector<std::string>, BatchCompletion, std::stop_token)>;

  EmbeddingModel() = default;
  explicit EmbeddingModel(EmbedManyFunction embed_many,
                          EmbedManyAsyncFunction embed_many_async = {})
      : embed_many_(std::move(embed_many)), embed_many_async_(std::move(embed_many_async)) {}

  [[nodiscard]] explicit operator bool() const noexcept { return static_cast<bool>(embed_many_); }
  [[nodiscard]] bool supports_async() const noexcept {
    return static_cast<bool>(embed_many_async_);
  }

  [[nodiscard]] Result<EmbeddingBatch> embed_many(const std::vector<std::string>& inputs,
                                                  std::stop_token stop = {}) const {
    if (!embed_many_)
      return std::unexpected(
          Error{.code = ErrorCode::invalid_configuration,
                .message = "The embedding model has no provider implementation."});
    if (auto valid = validate_inputs(inputs, stop); !valid)
      return std::unexpected(valid.error());
    return embed_many_(inputs, stop);
  }

  [[nodiscard]] Result<Embedding> embed(std::string_view input, std::stop_token stop = {}) const {
    return single_embedding(embed_many({std::string{input}}, stop));
  }

  [[nodiscard]] Result<void> embed_many_async(std::vector<std::string> inputs,
                                              BatchCompletion complete,
                                              std::stop_token stop = {}) const {
    if (!embed_many_async_ || !complete)
      return std::unexpected(
          Error{.code = ErrorCode::invalid_configuration,
                .message = "Async embeddings require a capable model and completion handler."});
    if (auto valid = validate_inputs(inputs, stop); !valid)
      return std::unexpected(valid.error());
    return embed_many_async_(std::move(inputs), std::move(complete), stop);
  }

  [[nodiscard]] Result<void> embed_async(std::string input, EmbeddingCompletion complete,
                                         std::stop_token stop = {}) const {
    if (!complete)
      return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                   .message = "Async embeddings require a completion handler."});
    return embed_many_async(
        {std::move(input)},
        [complete = std::move(complete)](Result<EmbeddingBatch> batch) {
          complete(single_embedding(std::move(batch)));
        },
        stop);
  }

private:
  [[nodiscard]] static Result<void> validate_inputs(const std::vector<std::string>& inputs,
                                                    std::stop_token stop) {
    if (stop.stop_requested())
      return std::unexpected(generation_cancelled_error());
    if (inputs.empty())
      return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                   .message = "Embedding requires at least one input."});
    for (const auto& input : inputs) {
      if (input.empty())
        return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                     .message = "Embedding inputs cannot be empty."});
    }
    return {};
  }

  [[nodiscard]] static Result<Embedding> single_embedding(Result<EmbeddingBatch> batch) {
    if (!batch)
      return std::unexpected(batch.error());
    if (batch->embeddings.size() != 1)
      return std::unexpected(
          Error{.code = ErrorCode::provider_response,
                .message = "The embedding provider returned the wrong number of vectors."});
    return std::move(batch->embeddings.front());
  }

  EmbedManyFunction embed_many_;
  EmbedManyAsyncFunction embed_many_async_;
};

} // namespace cail
