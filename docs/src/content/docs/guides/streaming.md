---
title: Streaming
description: Stream responses with the same prompts, tools, and options you use for text generation.
---

You can display model output as it arrives with `cail::stream_text`. It accepts
the same `GenerateTextOptions` as `generate_text`, plus an event callback.

Set `OPENAI_API_KEY` and run this example:

```cpp
#include <cail/generate.hpp>
#include <cail/openai.hpp>

#include <iostream>

int main() {
    auto response = cail::stream_text(
        {
            .model = cail::openai("gpt-6-luna"),
            .system = "You are a concise assistant.",
            .prompt = "Explain how streaming improves a chat interface.",
        },
        [](const cail::StreamEvent& event) {
            if (const auto* delta = std::get_if<cail::TextDelta>(&event)) {
                std::cout << delta->text << std::flush;
            }
        });
    if (!response) {
        std::cerr << response.error().message << '\n';
        return 1;
    }
    std::cout << '\n';
}
```

The call blocks until the response finishes and returns a complete
`GenerationResponse`. The callback runs synchronously as events arrive.
Streaming lets you display output earlier; it does not move work to another thread.

## Switch between generation and streaming

You can reuse your options for either call:

```cpp
cail::GenerateTextOptions options{
    .model = cail::openai("gpt-6-luna"),
    .system = "You are a concise assistant.",
    .prompt = "Summarize this paragraph.",
    .max_output_tokens = 256,
};

auto complete = cail::generate_text(options);
auto streamed = cail::stream_text(options, [](const cail::StreamEvent& event) {
    if (const auto* delta = std::get_if<cail::TextDelta>(&event)) {
        std::cout << delta->text << std::flush;
    }
});
```

These are two separate requests. Both accept `system`, `prompt`, `messages`,
`tools`, `structured_output`, `tool_loop`, `session_id`, `max_output_tokens`,
`stream_usage`, `provider_options`, and `middleware`.

When you supply both history and a prompt, `system` is prepended to the
messages and `prompt` is appended as a user message. You must supply a prompt
or at least one message; a system prompt alone is insufficient. Use `messages`
to include [images and PDFs](/guides/loaders/).

Add `.tools = {your_tool}` to stream a tool loop. The callback receives events
from every model step, and the returned response contains the final answer
and collected tool results. Use `tool_loop.max_rounds` and
`tool_loop.stop_when` to control when the loop ends. Middleware observes
requests and responses in the same way as with `generate_text`.

`structured_output` requests structured data when the model supports it. Events
still contain text deltas, not decoded partial C++ objects.

## Event types

The callback can receive:

- `TextDelta` for generated text
- `RefusalDelta` when the model refuses
- `ReasoningDelta` for reasoning summaries, when the provider reports them
- `ToolCallArgumentsDelta` and `ToolCallReady` for function tool calls
- `UsageUpdate` for token counts

Tool argument deltas and completed calls include an output index so you can
match them.

## Usage in the stream

Chat Completions streaming includes usage by default. Set `stream_usage` to
`false` when your endpoint does not support `stream_options.include_usage`.

When the provider reports them, `response->usage->cache_read_tokens` and
`reasoning_tokens` contain extra token counts. These fields are optional, so an
absent value means the provider did not report that detail.

## Cancel generation or streaming

Set `tool_loop.stop` to a `std::stop_token` for either `generate_text` or
`stream_text`. Request a stop from another thread, or from a streaming callback:

```cpp
std::stop_source stop;
auto response = cail::stream_text(
    {
        .model = cail::openai("gpt-6-luna"),
        .prompt = "Write a long story.",
        .tool_loop = {.stop = stop.get_token()},
    },
    [&](const cail::StreamEvent& event) {
        if (std::holds_alternative<cail::TextDelta>(event)) {
            stop.request_stop();
        }
    });
if (!response && response.error().code == cail::ErrorCode::cancelled) {
    std::cout << "Stopped.\n";
}
```

A token that is already stopped prevents the first model request. The token
also applies to subsequent model calls in a tool loop. Cancelling does not
interrupt a tool handler that is already running.

A streaming request retries only when no response data has reached your
callback, so a retry never duplicates an emitted chunk.

## Check streaming support

Check `model.adapter_capabilities().streaming` before streaming with a custom
model. `stream_text` returns an `invalid_configuration` error when the model
has no streaming implementation or the event handler is empty.

For manually managed requests, you can use `LanguageModel::stream()` or
`stream_tool_loop()` directly.

## Next steps

- [Tools](/guides/tools/) for a complete streaming tool example
- [Advanced usage](/guides/advanced/) for concurrency and timeouts
