---
title: Quickstart
description: Send a prompt, stream the reply, and decode a structured result with CAIL.
---

You can make your first CAIL request with a few lines of C++23. Set an API key
in your environment, pick a provider, and call the provider-neutral generation
helpers.

This guide uses OpenAI. The same calls work with every [provider](/guides/providers/)
CAIL supports.

## Prerequisites

- CAIL installed or on your include path, following [Install](/guides/install/)
- An API key for the provider you choose, for example `OPENAI_API_KEY`

## Send a prompt

The simplest call is `cail::generate_text`:

```cpp
#include <cail/cail.hpp>
#include <iostream>

int main()
{
    auto response = cail::generate_text({
        .model = cail::openai("gpt-6-luna"),
        .system = "You are a concise assistant.",
        .prompt = "Summarize the main idea of this paragraph.",
    });

    if (!response) {
        std::cerr << response.error().message << '\n';
        return 1;
    }
    std::cout << response->text << '\n';
}
```

`cail::openai` reads `OPENAI_API_KEY` from the environment. Requests time out
after 30 seconds and 429 and 5xx responses are retried twice with exponential
backoff. You can tune both when you create a model:

```cpp
auto model = cail::openai(
    "gpt-6-luna",
    cail::make_default_http_transport({
        .timeout = std::chrono::seconds{60},
        .retry = {.max_retries = 4},
    }));
```

Set `max_retries` to `0` to disable retries.

## Stream the reply

Call `LanguageModel::stream()` to receive events as they arrive. The call
returns the complete `GenerationResponse` after the provider finishes:

```cpp
auto response = cail::openai("gpt-6-luna").stream(
    "Summarize this paragraph.",
    [](const cail::StreamEvent& event) {
        if (const auto* delta = std::get_if<cail::TextDelta>(&event)) {
            std::cout << delta->text << std::flush;
        }
    });
```

See [Streaming](/guides/streaming/) for all event types and cancellation.

## Get a structured result

Describe a result with `cail::Field<T>` members and call `generate_object`:

```cpp
struct Answer {
    std::string language;
};

auto answer = cail::generate_object<Answer>({
    .model = cail::openai("gpt-6-luna"),
    .prompt = "What language is CAIL written in?",
});
if (answer) std::cout << answer->language << '\n';
```

The model needs JSON Schema support for structured output. See
[Structured outputs](/guides/structured-output/) for schema fields and
optionality.

## Next steps

- [Providers](/guides/providers/) for Anthropic, Gemini, OpenRouter, and local models
- [Request controls and results](/guides/request-controls/) for temperature, tool choice, and usage
- [Tools](/guides/tools/) to let the model call your functions
- [Agent](/guides/agent/) to bundle a model, instructions, and tools
- [Memory](/guides/memory/) to make an agent remember earlier turns
- [Files, images, and PDFs](/guides/loaders/) to attach local documents
- [Streaming](/guides/streaming/) for event types and cancellation
- [Async and coroutines](/guides/async/) to `co_await` calls from your application
- [Embeddings](/guides/embeddings/) and [EmbeddingStore](/guides/embeddings/#store-and-search-embeddings) for semantic search
