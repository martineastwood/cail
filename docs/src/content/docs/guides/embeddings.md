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

## Embed without blocking

Use `embed_async` for one input or `embed_many_async` for a batch. Set your
provider's API key, then start a request and handle its result in a callback:

```cpp
std::promise<void> finished;
auto done = finished.get_future();
std::stop_source stop;
auto model = cail::openai.embedding_model("text-embedding-3-small");

auto started = model.embed_many_async(
    {"A red apple", "A green pear"},
    [&finished](cail::Result<cail::EmbeddingBatch> result) {
        if (result) std::cout << result->embeddings.size() << " vectors\n";
        else std::cerr << result.error().message << '\n';
        finished.set_value();
    },
    stop.get_token());
if (!started) {
    std::cerr << started.error().message << '\n';
    return 1;
}
done.wait();
```

Include `<future>`, `<iostream>`, and `<stop_token>` alongside your CAIL
headers. This example waits to keep the process alive; your application can
continue other work after the call starts.

`embed_async` completes with `Result<Embedding>`. `embed_many_async` completes
with `Result<EmbeddingBatch>`, preserving input order. Immediate validation
errors are returned without invoking the callback. Once started, the operation
completes its callback once, including on cancellation or provider errors.

Built-in embedding models from CAIL providers generally support async requests,
including OpenAI-compatible presets, Gemini, Anthropic, and OpenCode. For custom
embedding models, check `supports_async()` first. The request keeps
the built-in model alive until completion; reference captures in callbacks must
remain valid. Callbacks may run on a CAIL I/O thread or before the call returns.

Call `stop.request_stop()` to cancel an active request or retry delay. Blocking
calls also accept a token: `embed(text, token)` and `embed_many(texts, token)`.

For a custom `EmbeddingModel`, the embedding function takes
`(const std::vector<std::string>& inputs, std::stop_token stop)`. Pass an optional
second function with `(std::vector<std::string> inputs, BatchCompletion complete,
std::stop_token stop)` to support async requests. Honor the token in both functions.

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

## Store and search embeddings

Use `cail::EmbeddingStore` to search a small collection of documents by meaning. Include
`<cail/embedding_store.hpp>`, add documents, then search with a query string:

```cpp
#include <cail/embedding_store.hpp>

cail::EmbeddingStore store(cail::openai.embedding_model("text-embedding-3-small"));
store.add({
    {.id = "apple", .text = "A red apple"},
    {.id = "pear", .text = "A green pear"},
    {.id = "banana", .text = "A yellow banana"},
});

auto results = store.search("A crunchy fruit", 2);
if (results) {
    for (const auto& result : *results) {
        std::cout << result.document.id << ' ' << result.score << '\n';
    }
}
```

The store embeds the documents in one batch and keeps them in memory. Each `search` embeds
the query and returns up to `top_k` documents (four by default), best match first. The
`score` is the cosine similarity between the query and the document, from -1 to 1.

`add` and `search` return an error rather than throwing, so check the result before you
read it. Searching an empty store returns no results without calling the provider.

The store compares every query against every document in memory, which is fast for small
collections and needs no database. Adding a document with an existing `id` replaces the
stored text and vector. Call `clear()` to remove every document and start over. For large
collections or persistence across restarts, use a dedicated vector database and keep the
model ID and dimension count alongside your vectors.

## Build and run the examples

```sh
cmake --build build --target cail_openai_embed cail_gemini_embed cail_mistral_embed cail_openai_embed_store
./build/cail_openai_embed_store
```

Set the matching environment variable first: `OPENAI_API_KEY`, `GEMINI_API_KEY`, or
`MISTRAL_API_KEY`.

## Next steps

- [Providers](/guides/providers/) to configure an embedding model for each provider
- [Advanced usage](/guides/advanced/) for middleware and per-request options
