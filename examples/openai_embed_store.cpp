#include <cail/cail.hpp>
#include <cail/embedding_store.hpp>

#include <iostream>
#include <vector>

int main() {
  cail::EmbeddingStore store(cail::openai.embedding_model("text-embedding-3-small"));
  if (const auto added = store.add({
          {.id = "apple", .text = "A red apple"},
          {.id = "pear", .text = "A green pear"},
          {.id = "banana", .text = "A yellow banana"},
      });
      !added) {
    std::cerr << added.error().message << '\n';
    return 1;
  }

  const auto results = store.search("A crunchy fruit", 2);
  if (!results) {
    std::cerr << results.error().message << '\n';
    return 1;
  }
  for (const auto& result : *results) {
    std::cout << result.document.id << ' ' << result.score << '\n';
  }
}
