#include <cail/cail.hpp>

#include <iostream>
#include <vector>

int main() {
  const auto model = cail::gemini.embedding_model("gemini-embedding-001");
  const auto one = model.embed("A red apple");
  if (!one) {
    std::cerr << one.error().message << '\n';
    return 1;
  }
  std::cout << one->model << ": " << one->dimensions << " dimensions\n";

  // Gemini accepts a shorter output size when you ask for one.
  const auto short_model = cail::gemini.embedding_model("gemini-embedding-001", 768);
  const auto batch = short_model.embed_many({"A red apple", "A green pear"});
  if (!batch) {
    std::cerr << batch.error().message << '\n';
    return 1;
  }
  std::cout << batch->model << ": " << batch->embeddings.size() << " vectors, " << batch->dimensions
            << " dimensions each\n";
}
