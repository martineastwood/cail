#include "test_support.hpp"

#include <cail/embedding_store.hpp>

#include <cmath>
#include <string>
#include <vector>

namespace test {

// "cat" and "feline" share a direction, "kitten" is close, "car" is orthogonal.
[[nodiscard]] std::vector<float> store_vector_for(const std::string& text) {
  if (text == "car") {
    return {0.0F, 1.0F};
  }
  if (text == "kitten") {
    return {0.9F, 0.1F};
  }
  return {1.0F, 0.0F};
}

void test_embedding_store_search() {
  std::size_t calls = 0;
  cail::EmbeddingStore store(cail::EmbeddingModel{
      [&calls](const std::vector<std::string>& inputs) -> cail::Result<cail::EmbeddingBatch> {
        ++calls;
        cail::EmbeddingBatch batch{.model = "stub", .dimensions = 2};
        for (const auto& input : inputs) {
          batch.embeddings.push_back(
              cail::Embedding{.values = store_vector_for(input), .model = "stub", .dimensions = 2});
        }
        return batch;
      }});

  check(store.empty(), "a new embedding store is empty");

  const auto added = store.add({
      {.id = "cat", .text = "cat"},
      {.id = "kitten", .text = "kitten"},
      {.id = "car", .text = "car"},
  });
  check(added.has_value() && store.size() == 3, "adding documents stores them");
  check(calls == 1, "adding documents embeds them in a single batch");

  const auto results = store.search("feline", 2);
  check(results.has_value() && results->size() == 2, "search returns at most top_k results");
  if (!results || results->size() != 2) {
    return;
  }
  check(results->at(0).document.id == "cat" && results->at(1).document.id == "kitten",
        "search orders documents by cosine similarity");
  check(std::abs(results->at(0).score - 1.0F) < 1e-6F &&
            results->at(0).score > results->at(1).score,
        "search reports the cosine similarity score");
  check(calls == 2, "search embeds the query once");

  const auto all = store.search("feline", 10);
  check(all.has_value() && all->size() == 3 && all->back().document.id == "car",
        "a top_k larger than the store returns every document, worst last");
}

void test_embedding_store_upsert_and_clear() {
  std::size_t calls = 0;
  cail::EmbeddingStore store(cail::EmbeddingModel{
      [&calls](const std::vector<std::string>& inputs) -> cail::Result<cail::EmbeddingBatch> {
        ++calls;
        cail::EmbeddingBatch batch{.model = "stub", .dimensions = 2};
        for (const auto& input : inputs) {
          batch.embeddings.push_back(
              cail::Embedding{.values = store_vector_for(input), .model = "stub", .dimensions = 2});
        }
        return batch;
      }});

  check(store.add({{.id = "cat", .text = "cat"}, {.id = "car", .text = "car"}}).has_value() &&
            store.size() == 2,
        "adding documents stores them");

  check(store.add({{.id = "cat", .text = "feline"}}).has_value() && store.size() == 2,
        "adding a document with an existing id replaces it");
  check(calls == 2, "upserting a document embeds the new text");

  const auto calls_before_clear = calls;
  store.clear();
  check(store.empty(), "clear removes every document");
  const auto empty = store.search("feline");
  check(empty.has_value() && empty->empty() && calls == calls_before_clear,
        "searching after clear returns nothing without calling the model");
}

void test_embedding_store_empty_and_errors() {
  std::size_t calls = 0;
  cail::EmbeddingModel model{
      [&calls](const std::vector<std::string>&) -> cail::Result<cail::EmbeddingBatch> {
        ++calls;
        return std::unexpected(
            cail::Error{.code = cail::ErrorCode::provider_response, .message = "boom"});
      }};

  cail::EmbeddingStore store(model);
  const auto empty = store.search("anything");
  check(empty.has_value() && empty->empty() && calls == 0,
        "searching an empty store returns nothing without calling the model");

  const auto added = store.add({{.id = "a", .text = "text"}});
  check(!added && added.error().message == "boom" && store.empty(),
        "adding documents propagates embedding errors and stores nothing");

  const auto zero = store.search("anything", 0);
  check(zero.has_value() && zero->empty(), "a top_k of zero returns no results");
}

void test_embedding_store_dimension_mismatch() {
  std::size_t calls = 0;
  cail::EmbeddingStore store(cail::EmbeddingModel{
      [&calls](const std::vector<std::string>& inputs) -> cail::Result<cail::EmbeddingBatch> {
        const std::size_t dimensions = calls == 0 ? 2 : 3;
        ++calls;
        cail::EmbeddingBatch batch{.model = "stub", .dimensions = dimensions};
        for (std::size_t i = 0; i < inputs.size(); ++i) {
          batch.embeddings.push_back(cail::Embedding{.values = std::vector<float>(dimensions, 1.0F),
                                                     .model = "stub",
                                                     .dimensions = dimensions});
        }
        return batch;
      }});

  check(store.add({{.id = "a", .text = "text"}}).has_value(), "a store accepts a document");

  const auto mixed_batch = store.add({{.id = "b", .text = "more"}});
  check(!mixed_batch && mixed_batch.error().code == cail::ErrorCode::invalid_configuration,
        "add rejects a batch whose dimensions do not match the store");

  bool query_is_three = true;
  cail::EmbeddingStore search_store(cail::EmbeddingModel{
      [&query_is_three](const std::vector<std::string>& inputs)
          -> cail::Result<cail::EmbeddingBatch> {
        const std::size_t dimensions = query_is_three && inputs.size() == 1 ? 3 : 2;
        query_is_three = false;
        cail::EmbeddingBatch batch{.model = "stub", .dimensions = dimensions};
        for (std::size_t i = 0; i < inputs.size(); ++i) {
          batch.embeddings.push_back(cail::Embedding{.values = std::vector<float>(dimensions, 1.0F),
                                                     .model = "stub",
                                                     .dimensions = dimensions});
        }
        return batch;
      }});
  check(search_store.add({{.id = "a", .text = "text"}}).has_value(),
        "a second store accepts a document");
  const auto results = search_store.search("query");
  check(!results && results.error().code == cail::ErrorCode::invalid_configuration,
        "search rejects a query whose dimensions do not match the stored documents");
}

} // namespace test

int main() {
  test::test_embedding_store_search();
  test::test_embedding_store_upsert_and_clear();
  test::test_embedding_store_empty_and_errors();
  test::test_embedding_store_dimension_mismatch();
  return test::failures == 0 ? 0 : 1;
}
