#pragma once

#include <cail/embedding_model.hpp>
#include <cail/error.hpp>

#include <cmath>
#include <cstddef>
#include <ranges>
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
  if (a.size() != b.size()) {
    return 0.0F;
  }
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

// An in-memory embeddings store. Add documents, then search by text with cosine
// similarity. Small and dependency-free: upsert by id to refresh a document, or
// call clear() to start over.
class EmbeddingStore {
public:
  explicit EmbeddingStore(EmbeddingModel model) : model_(std::move(model)) {}

  // Removes every document and resets dimension tracking.
  void clear() noexcept {
    entries_.clear();
    dimensions_ = 0;
  }

  // Embeds the documents in a single batch. Replaces an existing document with the
  // same id, otherwise appends.
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
    if (dimensions_ != 0 && batch->dimensions != dimensions_) {
      return std::unexpected(Error{
          .code = ErrorCode::invalid_configuration,
          .message = "The document embedding dimensions do not match the store.",
      });
    }
    if (batch->embeddings.size() != documents.size() || batch->dimensions == 0) {
      return std::unexpected(Error{
          .code = ErrorCode::provider_response,
          .message = "The embedding provider returned an invalid batch size or dimensions.",
      });
    }
    for (const auto& embedding : batch->embeddings) {
      if (embedding.values.size() != batch->dimensions ||
          embedding.dimensions != batch->dimensions) {
        return std::unexpected(Error{
            .code = ErrorCode::provider_response,
            .message = "The embedding provider returned a vector with the wrong length.",
        });
      }
    }
    dimensions_ = batch->dimensions;
    for (std::size_t i = 0; i < documents.size(); ++i) {
      const auto& id = documents[i].id;
      const auto existing = std::ranges::find_if(
          entries_, [&](const Entry& entry) { return entry.document.id == id; });
      if (existing != entries_.end()) {
        existing->document = std::move(documents[i]);
        existing->embedding = std::move(batch->embeddings[i]);
      } else {
        entries_.push_back(Entry{.document = std::move(documents[i]),
                                 .embedding = std::move(batch->embeddings[i])});
      }
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
    if (embedding->dimensions != dimensions_ || embedding->values.size() != dimensions_) {
      return std::unexpected(Error{
          .code = ErrorCode::invalid_configuration,
          .message = "The query embedding dimensions do not match the stored documents.",
      });
    }
    std::vector<SearchResult> results;
    results.reserve(entries_.size());
    for (const auto& entry : entries_) {
      if (entry.embedding.values.size() != dimensions_) {
        return std::unexpected(Error{
            .code = ErrorCode::provider_response,
            .message = "A stored embedding has the wrong vector length.",
        });
      }
      results.push_back(SearchResult{
          .document = entry.document,
          .score = detail::cosine_similarity(entry.embedding.values, embedding->values),
      });
    }
    std::ranges::sort(results, std::ranges::greater{}, &SearchResult::score);
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
  std::size_t dimensions_{};
};

} // namespace cail
