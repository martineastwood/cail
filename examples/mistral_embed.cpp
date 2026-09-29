#include <cail/cail.hpp>

#include <iostream>
#include <vector>

int main() {
  // For Mistral, OpenRouter, Ollama Cloud, and local servers, the endpoint is
  // derived from the provider's chat endpoint.
  const auto model = cail::mistral.embedding_model("mistral-embed");
  const auto batch = model.embed_many({"A red apple", "A green pear"});
  if (!batch) {
    std::cerr << batch.error().message << '\n';
    return 1;
  }
  std::cout << batch->model << ": " << batch->embeddings.size() << " vectors, " << batch->dimensions
            << " dimensions each\n";
}
