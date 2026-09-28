---
title: Streaming
description: Receive text, reasoning, tool-call, and usage events as the provider generates them.
---

You can receive a response as it is generated instead of waiting for the whole
reply. Call `LanguageModel::stream()` with a callback that receives
`cail::StreamEvent`s. The call returns the complete `GenerationResponse` after
the provider finishes.

```cpp
auto response = cail::openai("gpt-6-luna").stream(
    "Summarize this paragraph.",
    [](const cail::StreamEvent& event) {
        if (const auto* delta = std::get_if<cail::TextDelta>(&event)) {
            std::cout << delta->text << std::flush;
        }
    });
```

The callback runs synchronously while `stream()` processes the response. The
returned result reports success or failure and includes the final text,
reasoning, tool calls, and usage.

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

## Cancel a stream

Pass a `std::stop_token` and request a stop from another thread or from the
callback:

```cpp
std::stop_source stop;
auto response = cail::openai("gpt-6-luna").stream(
    "Write a long story.",
    [&](const cail::StreamEvent& event) {
        if (std::holds_alternative<cail::TextDelta>(event)) {
            stop.request_stop();
        }
    },
    stop.get_token());
// response.error().code is cail::ErrorCode::cancelled when stopped.
```

A streaming request retries only when no response data has reached your
callback, so a retry never duplicates an emitted chunk.

## Next steps

- [Tools](/guides/tools/) for streaming a tool loop
- [Advanced usage](/guides/advanced/) for concurrency and timeouts
