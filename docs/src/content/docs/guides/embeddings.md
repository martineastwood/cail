---
title: Embeddings
description: Turn text into vectors for search or similarity with CAIL's embedding models.
---

You can turn text into vectors for search or similarity. Build an embedding model from any
provider (provided their models support embeddings), then embed one text or a batch:

```cpp
auto model = cail::openai.embedding_model("text-embedding-3-small");
auto one = model.embed("A red apple");
auto batch = model.embed_many({"A red apple", "A green pear"});
if (batch) {
    std::cout << batch->model << ": " << batch->dimensions << " dimensions\n";
    // batch->embeddings[i] matches input i.
}
```

Every provider exposes an `embedding_model`. Pass the embedding model ID your provider
documents, the same way you pass a chat model ID to the provider itself. CAIL ships no
model catalog, so the lookup stays with you.

`embed` returns a single `Embedding`, and `embed_many` returns an `EmbeddingBatch`. Both
return an error rather than throwing, so check the result before you read the vectors.

Each vector includes the returned model ID and dimension count. Batch results follow
input order even if the provider returns indexed vectors out of order.
`batch->input_tokens` is present when the provider reports usage.

## Shorter vectors

The second argument is optional. Pass a positive dimension count when you want shorter
vectors from a model that supports them:

```cpp
auto model = cail::openai.embedding_model("text-embedding-3-small", 512);
```

Use the same model and dimensions for vectors you compare or store together. A vector
built at one length is not comparable with a vector built at another.

## Point a provider at your own endpoint

Embeddings follow the base URL you configured for the provider, so pointing a provider at
a gateway or proxy moves both chat and embeddings:

```cpp
auto mistral = cail::create_mistral({
    .endpoint = "https://proxy.internal/v1/chat/completions",
});
auto model = mistral.embedding_model("mistral-embed"); // https://proxy.internal/v1/embeddings
```

Servers that need no API key, such as `cail::local`, send no `Authorization` header.

Azure Foundry is configured per deployment, so pass the embeddings URL explicitly:

```cpp
auto model = cail::foundry.embedding_model("text-embedding-3-small", embeddings_url);
```

## Build and run the examples

```sh
cmake --build build --target cail_openai_embed cail_gemini_embed cail_mistral_embed
./build/cail_openai_embed
```

Set the matching environment variable first: `OPENAI_API_KEY`, `GEMINI_API_KEY`, or
`MISTRAL_API_KEY`.

## Next steps

CAIL does not ship a vector store. Store the vectors you get back in whatever database or
index you already use, and keep the model ID and dimension count alongside them.
