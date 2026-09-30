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

For sequential async workflows, use `co_await` with the
[coroutine interface](/guides/async/). For integrations that need callbacks,
you can start generation and receive the result in a completion callback.
`generate_text_async` accepts the same options as `generate_text`, including
system prompts, messages, tools, middleware, and cancellation.

Set `OPENAI_API_KEY` and run this example:

```cpp
#include <cail/cail.hpp>

#include <future>
#include <iostream>

int main() {
    std::promise<void> finished;
    auto done = finished.get_future();
    std::stop_source stop;

    auto started = cail::generate_text_async(
        {
            .model = cail::openai("gpt-6-luna"),
            .prompt = "Explain RAII in one sentence.",
            .tool_loop = {.stop = stop.get_token()},
        },
        [&finished](cail::Result<cail::GenerationResponse> result) {
            if (result) std::cout << result->text << '\n';
            else std::cerr << result.error().message << '\n';
            finished.set_value();
        });

    if (!started) {
        std::cerr << started.error().message << '\n';
        return 1;
    }
    done.wait(); // Keep this example alive until the callback finishes.
}
```

The return value reports whether the operation started. An immediate validation
or cancellation error is returned without invoking the callback. After an
operation starts, its completion callback receives one result: either a response
or an error. With tools, completion happens after the final model step.

Built-in providers support async generation through OpenAI Responses, Chat
Completions, Anthropic Messages, and Gemini. Azure Foundry and OpenCode use the
same async APIs. For a custom model, check
`model.adapter_capabilities().async_generation`. A custom HTTP transport must
implement `send_async` for async requests and `stream_async` for async streaming.

Requests own the model and input values until completion. Keep objects captured
by reference in callbacks, middleware, or tools alive until completion. Callbacks
can run before the initiating function returns, and may run on a CAIL I/O thread.
Cancellation can deliver completion on the thread that calls `request_stop`.
Keep callbacks brief and synchronize access to shared application state. Do not
wait for another async completion inside a callback. Blocking HTTP requests
inside an I/O callback return an `invalid_configuration` error; start an async
request instead.

Handle errors inside your completion callback. CAIL contains exceptions thrown
by terminal completion callbacks and never calls them again. Exceptions from
stream event handlers end the stream with an error. Synchronous tool handlers
run separately from I/O callbacks. See [Tools](/guides/tools/) for concurrency
limits and async tool handlers.

Call `stop.request_stop()` to cancel. The token applies to active HTTP requests,
retry delays, and later tool-loop steps. Tool handlers receive it through
`ToolContext::stop`, so they can stop cooperatively. Cancellation prevents the
remaining tools from starting after an active handler returns.

### Other async calls

| Task | Async call | Cancellation |
| --- | --- | --- |
| Text and tool loops | `generate_text_async(options, callback)` | `options.tool_loop.stop` |
| Typed structured output | `generate_object_async<T>(options, callback)` | `options.tool_loop.stop` |
| Text streams | `stream_text_async(options, on_event, callback)` | `options.tool_loop.stop` |
| Agent streams | `agent.stream_async(input, on_event, callback, options)` | `options.stop` |
| Agent prompts or requests | `agent.generate_async(input, callback, options)` | `options.stop` |
| One embedding | `model.embed_async(text, callback, token)` | Third argument |
| Embedding batch | `model.embed_many_async(texts, callback, token)` | Third argument |
| Manually managed generation | `model.generate_async(request, callback, token)` | Third argument |

`generate_object_async<T>` completes with `Result<T>` and reports the same
refusal, incomplete response, and JSON decoding errors as `generate_object<T>`.
Check `embedding_model.supports_async()` before async embedding calls with a
custom model.

Async agent text prompts and user messages load memory before starting the request and save successful
turns before completion. Memory loads and writes run asynchronously, including
file storage. An explicit
`GenerationRequest` bypasses memory, as with blocking agent calls.

Use `stream_text_async` or `Agent::stream_async` to stream without blocking your
caller. Built-in providers use native async streaming. Check
`model.adapter_capabilities().async_streaming` for custom models.

### Choose where callbacks run

You can deliver callbacks through your application's scheduler with
`ToolLoopOptions::async.schedule`. For model and embedding calls, pass
`cail::AsyncOptions` after the stop token. Without a scheduler, callbacks run on
the completing thread, possibly before the initiating call returns.

See [Streaming](/guides/streaming/#schedule-stream-callbacks) for a complete
example, event queue limits, and cancellation behavior.

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

See [Request controls and results](/guides/request-controls/) for sampling, stop
sequences, tool choice, finish reasons, and usage across model steps.

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
