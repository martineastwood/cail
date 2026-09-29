#pragma once

#include <cail/embedding_model.hpp>
#include <cail/error.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cail {

// A piece of text with a stable id, the unit an embedding store indexes.
struct Document {
  std::string id;
  std::string text;
};

// A document and its cosine similarity to a query, in [-1, 1].
struct SearchResult {
  Document document;
  float score{};
};

namespace detail {

// Cosine similarity of two equal-length vectors, 0 when either has no magnitude.
[[nodiscard]] inline float cosine_similarity(const std::vector<float>& a,
                                             const std::vector<float>& b) {
  float dot = 0.0F;
  float norm_a = 0.0F;
  float norm_b = 0.0F;
  for (std::size_t i = 0; i < a.size(); ++i) {
    dot += a[i] * b[i];
    norm_a += a[i] * a[i];
    norm_b += b[i] * b[i];
  }
  const float denominator = std::sqrt(norm_a) * std::sqrt(norm_b);
  return denominator == 0.0F ? 0.0F : dot / denominator;
}

} // namespace detail

// An in-memory embeddings store. Add documents once, then search by text with
// cosine similarity. Small and dependency-free: re-add and search again when
// your documents change.
class EmbeddingStore {
public:
  explicit EmbeddingStore(EmbeddingModel model) : model_(std::move(model)) {}

  // Embeds the documents in a single batch and keeps them for search.
  [[nodiscard]] Result<void> add(std::vector<Document> documents) {
    if (documents.empty()) {
      return {};
    }
    std::vector<std::string> texts;
    texts.reserve(documents.size());
    for (const auto& document : documents) {
      texts.push_back(document.text);
    }
    auto batch = model_.embed_many(texts);
    if (!batch) {
      return std::unexpected(batch.error());
    }
    for (std::size_t i = 0; i < documents.size(); ++i) {
      entries_.push_back(
          Entry{.document = std::move(documents[i]), .embedding = std::move(batch->embeddings[i])});
    }
    return {};
  }

  // Embeds the query and returns the closest documents, best first.
  [[nodiscard]] Result<std::vector<SearchResult>> search(std::string_view query,
                                                         std::size_t top_k = 4) const {
    if (entries_.empty() || top_k == 0) {
      return std::vector<SearchResult>{};
    }
    auto embedding = model_.embed(query);
    if (!embedding) {
      return std::unexpected(embedding.error());
    }
    std::vector<SearchResult> results;
    results.reserve(entries_.size());
    for (const auto& entry : entries_) {
      if (entry.embedding.dimensions != embedding->dimensions) {
        return std::unexpected(Error{
            .code = ErrorCode::invalid_configuration,
            .message = "The query embedding dimensions do not match the stored documents.",
        });
      }
      results.push_back(SearchResult{
          .document = entry.document,
          .score = detail::cosine_similarity(entry.embedding.values, embedding->values),
      });
    }
    std::sort(results.begin(), results.end(),
              [](const SearchResult& left, const SearchResult& right) {
                return left.score > right.score;
              });
    if (results.size() > top_k) {
      results.resize(top_k);
    }
    return results;
  }

  [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }

  [[nodiscard]] bool empty() const noexcept { return entries_.empty(); }

private:
  struct Entry {
    Document document;
    Embedding embedding;
  };

  EmbeddingModel model_;
  std::vector<Entry> entries_;
};

} // namespace cail
