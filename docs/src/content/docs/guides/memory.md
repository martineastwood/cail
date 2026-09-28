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

Each file holds the full message history for one conversation. You can inspect
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

Raw history grows without bound, and long histories cost tokens and distract
the model. Set `keep_last_messages` to send only the most recent messages.
Leading system and developer messages always stay, and trimming only shapes
what is sent: the backend keeps the full history, so nothing is lost:

```cpp
auto response = agent.generate("Continue.", {.keep_last_messages = 20});
```

Trailing tool results are kept together with the tool call that requested
them, so providers never see an unpaired result.

## Bypass memory for one call

Passing a `GenerationRequest` instead of a prompt string bypasses memory for
that call: nothing is loaded and nothing is stored. Use it when you want to
send exactly the history you choose:

```cpp
auto response = agent.generate(cail::GenerationRequest{
    .messages = {cail::Message{
        .content = {cail::TextPart{.text = "One-off question."}}},
    },
});
```

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
`append` receives the messages from one turn: the user prompt, the assistant
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
  very long conversations get slower to append to over time. `keep_last_messages`
  bounds what is sent, not what is stored, so the file keeps growing for the life
  of the conversation. Call `clear` when you want to start over.
- Compaction such as LLM-generated rolling summaries is not built in yet. You
  can implement it in a `ConversationMemory` wrapper of your own.

## Next steps

- [Agent](/guides/agent/) to attach a memory backend to an agent
- [Advanced usage](/guides/advanced/) for middleware and per-request options
