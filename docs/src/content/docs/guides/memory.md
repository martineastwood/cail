---
title: Memory
description: Give an agent conversation memory that loads, stores, and persists history per conversation.
---

You can give an agent conversation memory so it remembers earlier turns. Attach
a memory backend and a conversation id, and the agent loads the stored history
before each prompt and appends the new turn, including tool calls and their
results, after it succeeds.

```cpp
#include <cail/agent.hpp>
#include <cail/memory.hpp>
#include <cail/openai.hpp>

#include <iostream>
#include <memory>

int main() {
  cail::Agent agent({
      .model = cail::openai("gpt-6-luna"),
      .instructions = "You are a concise research assistant.",
      .memory = std::make_shared<cail::InMemoryConversationMemory>(),
      .conversation_id = "user-42",
  });

  agent.generate("My name is Ada.");
  auto response = agent.generate("What is my name?");
  if (!response) {
    std::cerr << response.error().message << '\n';
    return 1;
  }
  std::cout << response->text << '\n';
}
```

Build and run the example with `OPENAI_API_KEY` set:

```sh
cmake --build build --target cail_openai_memory
./build/cail_openai_memory
```

## Remember images and PDFs

Pass a `cail::Message` to include attachments in a conversation. The agent stores
the complete user message after a successful turn, so your next text prompt can
refer to the same image or document:

```cpp
auto pdf = cail::load_pdf("report.pdf");
auto image = cail::load_image("chart.png");
if (!pdf || !image) {
    std::cerr << (!pdf ? pdf.error().message : image.error().message) << '\n';
    return 1;
}

cail::Agent agent({
    .model = cail::openai("gpt-6-luna"),
    .memory = std::make_shared<cail::FileConversationMemory>("conversations"),
    .conversation_id = "report-review",
});

auto first = agent.generate(cail::Message{
    .content = {
        cail::TextPart{.text = "Compare this chart with the report's findings."},
        std::move(*image),
        std::move(*pdf),
    },
});
if (!first) {
    std::cerr << first.error().message << '\n';
    return 1;
}
auto follow_up = agent.generate("Which finding best explains the chart?");
if (!follow_up) {
    std::cerr << follow_up.error().message << '\n';
    return 1;
}
std::cout << follow_up->text << '\n';
```

Include `<cail/loaders.hpp>` and `<utility>` alongside the headers in the
quickstart. You can run the complete example from a checkout with
`OPENAI_API_KEY` set and your own PDF and image:

```sh
cmake --build build --target cail_openai_multimodal_memory
./build/cail_openai_multimodal_memory report.pdf chart.png
```

Both memory backends preserve attachment bytes, MIME types, PDF filenames,
content order, and provider options. File memory keeps a copy of the attachments,
so follow-ups do not require the original files. Recreate an agent with the same
memory directory and conversation id to continue after restarting your application.

A single-message input must have the `user` role, at least one content part,
and no tool calls or tool call ID. For full conversation history, pass a
`GenerationRequest` instead. Your provider and model must accept the attachments;
see [Files, images, and PDFs](/guides/loaders/) for supported formats.

### Stream or run asynchronously

The same user message works with `agent.stream(message, callback, options)` and
`agent.generate_async(message, callback, options)`. Use
`agent.stream_async(message, on_event, complete, options)` for async streaming:

```cpp
cail::Message message{
    .content = {cail::TextPart{.text = "Summarize the report's main conclusion."}},
};
auto started = agent.generate_async(
    std::move(message),
    [](cail::Result<cail::GenerationResponse> result) {
        if (result) std::cout << result->text << '\n';
        else std::cerr << result.error().message << '\n';
    });
if (!started) {
    std::cerr << started.error().message << '\n';
}
```

This follow-up reuses the report and chart already in memory. Keep your application
running until the callback finishes. Successful turns are saved before completion;
generation failures store no new attachments. Async calls load history and
save successful turns without blocking your caller. Cancellation stops queued
memory work, but a write already running may finish. Callback delivery errors
do not roll back a saved turn. Conversation overrides,
`keep_last_turns`, tools, and cancellation work the same way as for text prompts.

For a remote store, you can override `load_async`, `append_async`, and
`clear_async` on `ConversationMemory`. Each returns `Result<void>` to report
whether it started, takes a completion callback and stop token, and completes
once after successful initiation. Async operations must own their inputs until
completion and permit concurrent calls. If you implement only the synchronous
methods, CAIL runs them on bounded workers. Use shared ownership for async stores.

## Pick a backend

CAIL ships two implementations of the `cail::ConversationMemory` interface:

- `cail::InMemoryConversationMemory` keeps history in process memory. It is
  ideal for tests and short-lived agents, and it forgets everything on restart.
- `cail::FileConversationMemory` stores one JSON file per conversation id in a
  directory you choose. History survives a restart, so this is the durable
  option:

```cpp
cail::Agent agent({
    .model = cail::openai("gpt-6-luna"),
    .memory = std::make_shared<cail::FileConversationMemory>("conversations"),
    .conversation_id = "user-42",
});
```

Each file holds the full message history for one conversation. Binary attachments
are stored as base64 within the JSON. You can inspect
or back up the files directly, and `clear` deletes the conversation's file.

## Set the conversation id

Each conversation id keeps its own isolated history. Set a default on the
agent, or override it per call:

```cpp
cail::Agent agent({
    .model = cail::openai("gpt-6-luna"),
    .memory = std::make_shared<cail::InMemoryConversationMemory>(),
    .conversation_id = "user-42",
});

agent.generate("Hello from user 42.");                       // uses the default id
agent.generate("Hello from user 43.", {.conversation_id = "user-43"});
```

A call with no conversation id anywhere still works. It just skips memory:
nothing is loaded and nothing is stored.

## Keep the window small

Stored images and PDFs are sent again on follow-up requests while their original
message remains in the window. This increases request size and can reach your
provider's input limits. Trimming a message drops all of its attachments from
the request, while the full attachments remain in storage.

Raw history grows without bound, and long histories cost tokens and distract
the model. Set `keep_last_turns` to send only the most recent user turns.
Leading system and developer messages always stay, and trimming only shapes
what is sent: the backend keeps the full history, so nothing is lost. Use
`MessageRole::developer` for provider-specific instructions that should stay at
the top of the window with system messages:

```cpp
auto response = agent.generate("Continue.", {.keep_last_turns = 10});
```

A turn starts with a user message and includes every assistant reply and tool
exchange until the next user message. The limit counts previous turns; your
current prompt is added afterward. Set it to `0` to send the full history.

Trimming removes whole turns, including their attachments, tool calls, and tool
results. It does not limit tokens or the number of messages within a turn.

`keep_last_turns` replaces `keep_last_messages`: update your calls to use a turn
count when upgrading.

## Bypass memory for one call

Passing a `GenerationRequest` instead of a text prompt or single message bypasses memory for
that call: nothing is loaded and nothing is stored. Use it when you want to
send exactly the history you choose:

```cpp
auto response = agent.generate(cail::GenerationRequest{
    .messages = {cail::Message{
        .content = {cail::TextPart{.text = "One-off question."}}},
    },
});
```

## Reset an older conversation file

File memory now uses format version 2 to store binary attachments. Version 1
files return an unsupported-version error. Clear a conversation before reusing
its id with the new format:

```cpp
auto memory = std::make_shared<cail::FileConversationMemory>("conversations");
auto cleared = memory->clear("report-review");
if (!cleared) {
    std::cerr << cleared.error().message << '\n';
    return 1;
}
```

`clear` removes the whole stored conversation, including its attachments. Back up
any history you need before clearing it.

## Store history in your own database

Implement the `cail::ConversationMemory` interface over your own store. It has
three methods:

```cpp
class MyConversationMemory final : public cail::ConversationMemory {
public:
  cail::Result<std::vector<cail::Message>>
  load(const std::string& conversation_id) override;

  cail::Result<void> append(const std::string& conversation_id,
                            std::vector<cail::Message> messages) override;

  cail::Result<void> clear(const std::string& conversation_id) override;
};
```

`load` returns an empty vector for a conversation that has no history yet.
`append` receives the messages from one turn: the complete user message, the assistant
reply, and every tool round in between. A failed `generate` call stores
nothing, so history only ever contains completed turns.

## Read the turn from a response

Calls that run the tool loop, such as `agent.generate` and
`cail::generate_text`, report the messages the call added in `turn`, whether or
not the agent uses memory. You can record turns with your own storage:

```cpp
auto response = agent.generate("What is the weather in Paris?");
history.insert(history.end(), response->turn.begin(), response->turn.end());
```

## Limitations

- A memory backend is not synchronized across threads. Give each thread its
  own agent and backend, or guard shared backends yourself.
- `FileConversationMemory` rewrites the conversation file on every append, so
  very long conversations get slower to append to over time. `keep_last_turns`
  bounds what is sent, not what is stored, so the file keeps growing for the life
  of the conversation. Call `clear` when you want to start over.
- Compaction such as LLM-generated rolling summaries is not built in yet. You
  can implement it in a `ConversationMemory` wrapper of your own.

## Use memory with concurrent requests

Share one memory instance between agents that use the same conversation.
A memory-backed turn rejects a second turn for the same conversation with a
`memory` error until the first finishes. A turn paused for approval stays reserved
until you finish it through `agent.resume` or discard every copy of its continuation.
Only the completed turn is saved. See
[Remember approved tool calls](/guides/tools/#remember-approved-tool-calls).
Wait for the generation completion callback before sending the next message. Different conversation IDs can run
concurrently.

The built-in memory stores support concurrent loads, appends, and clears in one
process. File memory does not coordinate writes from separate processes. If you
supply your own memory store, make its operations safe for concurrent calls.

## Next steps

- [Agent](/guides/agent/) to attach a memory backend to an agent
- [Files, images, and PDFs](/guides/loaders/) to load local attachments
- [Advanced usage](/guides/advanced/) for middleware and per-request options
