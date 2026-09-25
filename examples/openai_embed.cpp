#include <cail/cail.hpp>

#include <iostream>
#include <vector>

int main()
{
    const auto model = cail::openai.embedding_model("text-embedding-3-small");
    const auto batch = model.embed_many({"A red apple", "A green pear"});
    if (!batch) {
        std::cerr << batch.error().message << '\n';
        return 1;
    }
    std::cout << batch->model << ": " << batch->embeddings.size()
              << " vectors, " << batch->dimensions << " dimensions each\n";
}
