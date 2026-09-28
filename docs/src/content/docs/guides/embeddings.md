---
title: Embeddings
description: Turn text into vectors for search or similarity with CAIL's embedding models.
---

You can turn text into vectors for search or similarity. With
`OPENAI_API_KEY` set, create an embedding model and embed one text or a batch:

```cpp
auto model = cail::openai.embedding_model("text-embedding-3-small");
auto one = model.embed("A red apple");
auto batch = model.embed_many({"A red apple", "A green pear"});
if (batch) {
    std::cout << batch->model << ": " << batch->dimensions << " dimensions\n";
    // batch->embeddings[i] matches input i.
}
```

Each vector includes the returned model ID and dimension count. Batch results
follow input order even if the provider returns indexed vectors out of order.
`batch->input_tokens` is present when the provider reports usage.

## Shorter vectors

For OpenAI's `text-embedding-3` models, pass a positive dimension count as the
second argument to `embedding_model` when you want shorter vectors:

```cpp
auto model = cail::openai.embedding_model("text-embedding-3-small", 512);
```

Use the same model and dimensions for vectors you compare or store together.

## Build and run the example

```sh
cmake --build build --target cail_openai_embed
./build/cail_openai_embed
```
