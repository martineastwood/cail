---
title: Advanced usage
description: Concurrency, middleware, provider-specific options, images, and token limits in CAIL.
---

These patterns cover what you need once the basic calls work: moving requests
off your thread, tuning HTTP behavior, observing requests with middleware, and
reaching provider features beyond the common API.

## Run concurrent requests

`generate()`, `stream()`, embeddings, and tool loops block the calling thread.
You can schedule them on your application's executor and share a provider model
across concurrent tasks.

For example, you can run two requests with `std::async`:

```cpp
auto model = cail::openai("gpt-6-luna");
auto first = std::async(std::launch::async, [&] {
    return cail::generate_text({.model = model, .prompt = "Summarize document A."});
});
auto second = std::async(std::launch::async, [&] {
    return cail::generate_text({.model = model, .prompt = "Summarize document B."});
});

auto first_response = first.get();
auto second_response = second.get();
```

The default HTTP transport supports concurrent calls. If you supply your own
transport, it must also support calls from multiple threads.

Set `tool_loop.stop` to cancel a nonstreaming request, including a request
waiting to retry:

```cpp
std::stop_source stop;
auto pending = std::async(std::launch::async, [&] {
    return cail::generate_text({
        .model = model,
        .prompt = "Summarize this document.",
        .tool_loop = {.stop = stop.get_token()},
    });
});

stop.request_stop();
auto response = pending.get();
// response.error().code is cail::ErrorCode::cancelled when stopped.
```

For a direct model call, pass the token as the second argument to
`model.generate(request, stop.get_token())`.

## Generate without blocking

Chat Completions models can send a request and return immediately. Use
`generate_async()` when you want CAIL to call you after the response arrives.
This example assumes an OpenAI-compatible local server with a `llama3.2` model:

```cpp
#include <cail/cail.hpp>
#include <future>
#include <print>

int main() {
    auto model = cail::create_local({
        .endpoint = "http://localhost:11434/v1/chat/completions",
    })("llama3.2");
    std::promise<void> finished;
    auto done = finished.get_future();

    auto started = model.generate_async(
        {.messages = {cail::Message{
            .content = {cail::TextPart{.text = "Explain RAII in one sentence."}},
        }}},
        [&finished](cail::Result<cail::GenerationResponse> result) {
            if (result) std::println("{}", result->text);
            else std::println(stderr, "{}", result.error().message);
            finished.set_value();
        });

    if (!started) {
        std::println(stderr, "{}", started.error().message);
        return 1;
    }
    done.wait(); // Keep this small program alive until the callback finishes.
}
```

The return value reports whether the request started. Once it starts, the
completion callback receives either the response or an error, including
`ErrorCode::cancelled` if you request a stop. The callback may run on a CAIL I/O
thread, so pass work to your application's executor if it takes time. Keep the
callback free of uncaught exceptions.
Keep any objects captured by reference in the callback or middleware alive until
the callback finishes.

To cancel, pass `stop.get_token()` as the third argument and call
`stop.request_stop()`. A request that already started completes its callback
with `ErrorCode::cancelled`.

Check `model.adapter_capabilities().async_generation` before using this API
with a model. It is currently available for Chat Completions providers, such as
Local, OpenRouter, and Mistral. Other providers report that async generation is
unavailable. Async tool loops and streaming are not yet available. If you use a
custom HTTP transport for async calls, implement its `send_async()` operation.

## Timeouts and retries

Provider requests time out after 30 seconds and retry 429 and 5xx responses
twice with exponential backoff. Tune both when you create a model:

```cpp
auto model = cail::openai(
    "gpt-6-luna",
    cail::make_default_http_transport({
        .timeout = std::chrono::seconds{60},
        .retry = {.max_retries = 4},
    }));
```

Set `max_retries` to `0` to disable retries. Streaming requests retry only when
no response data has reached your callback, so a retry never duplicates an
emitted chunk.

## Middleware

Add middleware when you need headers, logging, metrics, or tracing around
generation:

```cpp
cail::GenerationMiddleware tracing{
    .before_request = [](cail::HttpRequest& request, const cail::MiddlewareContext& context) {
        request.headers.push_back({.name = "x-model-step", .value = std::to_string(context.step)});
        std::println("starting model step {}", context.step);
    },
    .after_response = [](const cail::HttpResponse& response,
                         const cail::MiddlewareContext& context) {
        std::println("step {} returned HTTP {}", context.step, response.status_code);
    },
    .after_step = [](const cail::Result<cail::GenerationResponse>& result,
                     const cail::MiddlewareContext& context) {
        std::println("step {} {}", context.step, result ? "completed" : "failed");
    },
};

auto response = cail::generate_text({
    .model = cail::openai("gpt-6-luna"),
    .prompt = "What is the capital of France?",
    .middleware = {tracing},
});
```

Middleware runs in registration order. `before_request` can mutate the encoded
HTTP request, `after_response` observes the raw response, and `after_step`
receives the decoded result. A tool loop increments `context.step` for every
follow-up model call.

You can use these hooks with your OpenTelemetry SDK to inject trace headers and
record HTTP status, errors, token usage, and tool-loop steps. CAIL does not
require an OpenTelemetry dependency.

## Limit output tokens

Set `max_output_tokens` to cap a generation. The limit also applies to tool
follow-up requests:

```cpp
auto response = cail::generate_text({
    .model = cail::openai("gpt-6-luna"),
    .prompt = "Summarize this paragraph in three sentences.",
    .max_output_tokens = 256,
});
```

## Provider-specific options

You can pass provider-specific JSON fields on a request or in message history
when a provider needs data beyond the common API. CAIL returns opaque response
metadata in `GenerationResponse::provider_options`, so you can save it with the
assistant message and send it back on a follow-up request:

```cpp
auto model = cail::openrouter("openai/gpt-4o-mini");
cail::GenerationRequest request{
    .messages = {cail::Message{
        .role = cail::MessageRole::user,
        .content = {cail::TextPart{.text = "What is CAIL?"}},
    }},
};
auto response = model.generate(request);
if (response) {
    request.messages.push_back(cail::Message{
        .role = cail::MessageRole::assistant,
        .content = {cail::TextPart{.text = response->text}},
        .provider_options = response->provider_options,
    });
    request.messages.push_back(cail::Message{
        .role = cail::MessageRole::user,
        .content = {cail::TextPart{.text = "Tell me one more thing."}},
    });
    auto follow_up = model.generate(request);
}
```

For Chat Completions providers, request options are added to the request body.
Message, content-part, and tool-definition options are added to their matching
history entries. Use only fields accepted by the provider you selected.

## Send an image

You can include text and image bytes in a user message when the selected OpenAI
model supports images:

```cpp
#include <cail/cail.hpp>

#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <utility>

int main()
{
    std::ifstream image("screenshot.png", std::ios::binary);
    if (!image) return 1;
    std::string bytes(std::istreambuf_iterator<char>{image}, {});

    auto response = cail::openai("gpt-6-luna").generate(cail::GenerationRequest{
        .messages = {cail::Message{
            .role = cail::MessageRole::user,
            .content = {
                cail::TextPart{.text = "What does this screenshot show?"},
                cail::ImagePart{.bytes = std::move(bytes), .mime_type = "image/png"},
            },
        }},
    });
    if (!response) {
        std::cerr << response.error().message << '\n';
        return 1;
    }
    std::cout << response->text << '\n';
}
```

Pass the original file bytes and the matching MIME type. Image parts are
supported in user and tool-result messages; assistant messages cannot contain
images. If you build tool-result history yourself, add an `ImagePart` to the
tool message content and CAIL encodes it for the selected provider.

## Errors

Errors include a stable `code`; OpenAI errors can also include `http_status`,
`provider_code`, `provider_type`, and `request_id`. A cancelled stream reports
`cail::ErrorCode::cancelled`.

## Next steps

- [Streaming](/guides/streaming/) for cancellation with stop tokens
- [Providers](/guides/providers/) for provider-specific setup
