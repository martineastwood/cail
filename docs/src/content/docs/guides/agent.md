---
title: Agent
description: Bundle a model, instructions, and tools into an agent that runs its own tool loop.
---

You can bundle a model, standing instructions, and a set of tools into a
`cail::Agent`, then run prompts against it. Each call runs the tool loop for
you: the agent sends its instructions as the first system message, offers its
tools, executes the tool calls, and returns the final response.

```cpp
#include <cail/agent.hpp>
#include <cail/openai.hpp>

#include <iostream>

int main() {
  cail::Agent agent({
      .model = cail::openai("gpt-6-luna"),
      .instructions = "You are a concise research assistant.",
  });
  auto response = agent.generate("Name one benefit of native C++ AI applications.");
  if (!response) {
    std::cerr << response.error().message << '\n';
    return 1;
  }
  std::cout << response->text << '\n';
}
```

Build and run the example with `OPENAI_API_KEY` set:

```sh
cmake --build build --target cail_openai_agent
./build/cail_openai_agent
```

## Configure the agent

`AgentConfig` takes five fields:

- `model`: any CAIL model, such as `cail::openai("gpt-6-luna")` or a model from
  `create_local`. See [Providers](/guides/providers/).
- `instructions`: standing instructions prepended to every request as the first
  system message. Leave it empty to send none.
- `tools`: the tools offered on every run. See [Tools](/guides/tools/) for
  creating typed handlers with `cail::tool`.
- `memory`: an optional conversation history backend. See
  [Memory](/guides/memory/).
- `conversation_id`: the default conversation id used with `memory`.

## Add tools

```cpp
struct WeatherQuery {
    std::string location;
};

struct WeatherReport {
    std::string location;
    int temperature_fahrenheit{};
};

auto weather_tool = cail::tool<WeatherQuery, WeatherReport>(
    "weather", "Get the weather in a location.",
    [](const WeatherQuery& query) {
        return WeatherReport{.location = query.location, .temperature_fahrenheit = 72};
    });

cail::Agent agent({
    .model = cail::openai("gpt-6-luna"),
    .instructions = "Answer questions about the weather.",
    .tools = {weather_tool},
});

auto response = agent.generate("Fetch the weather for Paris.");
```

When the model requests a tool call, CAIL decodes the arguments into your input
type, runs your handler, and sends the result back for the next model step. The
returned `GenerationResponse` contains the final text and each tool result.

## Pass a full request

For history, images, or a session ID, pass a `GenerationRequest` instead of a
prompt string. The agent still prepends its instructions and offers its tools:

```cpp
cail::GenerationRequest request{
    .messages = {cail::Message{
        .role = cail::MessageRole::user,
        .content = {cail::TextPart{.text = "What did we discuss earlier?"}},
    }},
    .session_id = "conversation-42",
};
auto response = agent.generate(request);
```

Passing a request also bypasses agent memory: nothing is loaded and nothing is
stored. See [Memory](/guides/memory/) to keep history for prompt calls.

## Tune the loop

Both `generate` overloads accept `ToolLoopOptions`:

```cpp
std::stop_source stop;
auto response = agent.generate(
    "Fetch the weather for Paris and Tokyo.",
    {
        .max_rounds = 4,
        .stop_when = [](const cail::GenerationResponse& response,
                        const cail::MiddlewareContext&) {
            return response.status == cail::GenerationStatus::refused;
        },
        .stop = stop.get_token(),
    });
```

- `max_rounds` caps the follow-up model calls. The default is 8.
- `stop_when` runs after a model step and before its requested tools execute;
  returning `true` returns that response without starting another step.
- `stop` cancels the active stream when you call `stop.request_stop()`.

## Stream a run

`agent.stream()` takes the same event callback as a model stream and runs the
tool loop while events arrive:

```cpp
auto response = agent.stream(
    "Fetch the weather for Paris.",
    [](const cail::StreamEvent& event) {
        if (const auto* delta = std::get_if<cail::TextDelta>(&event)) {
            std::cout << delta->text << std::flush;
        }
    },
    {.max_rounds = 4});
```

The callback receives text, reasoning, usage, and tool-call events from every
step. See [Streaming](/guides/streaming/) for the full event list.

## Manage the loop yourself

`Agent` is a convenience over the lower-level APIs. When you want to inspect or
drive each round yourself, call the model with `LanguageModel::generate()`, or
run `cail::run_tool_loop` and `cail::stream_tool_loop` directly as shown in
[Tools](/guides/tools/).

## Ask about an image or document

Pass a user `Message` when your prompt includes images or PDFs:

```cpp
auto image = cail::load_image("chart.png");
if (!image) {
    std::cerr << image.error().message << '\n';
    return 1;
}
auto response = agent.generate(cail::Message{
    .content = {cail::TextPart{.text = "Explain this chart."}, std::move(*image)},
});
```

Include `<cail/loaders.hpp>` and `<utility>`. You can include several text,
image, and PDF parts in the same message. When you configure memory and a
conversation id, successful turns save the full message and replay its
attachments in later calls. The same input works with `stream` and
`generate_async`. See [Memory](/guides/memory/) for a complete conversation example.

Single-message inputs accept only the user role and cannot contain tool calls or
a tool call ID. Use `GenerationRequest` when you manage the full history yourself.

## Generate without blocking

Call `generate_async` with a prompt and completion callback:

```cpp
auto started = weather_agent.generate_async(
    "What should I pack for Paris?",
    [](cail::Result<cail::GenerationResponse> result) {
        if (result) std::cout << result->text << '\n';
        else std::cerr << result.error().message << '\n';
    });
if (!started) {
    std::cerr << started.error().message << '\n';
}
```

The callback receives the final result after any tool calls. Keep your application
running until it finishes. Text prompts and single user messages load configured memory before starting
and append successful turns before invoking the callback. Memory operations are
synchronous. Calls with an explicit `GenerationRequest` bypass memory.

Pass `ToolLoopOptions` as the third argument to set `max_rounds`, `stop_when`, or
`stop`. The request owns the model, tools, and memory backend until completion;
objects captured by reference in handlers must remain valid. Avoid overlapping
prompt calls for the same conversation because they can load the same history.
See [Advanced usage](/guides/advanced/) for async execution and cancellation.

## Next steps

- [Tools](/guides/tools/) for typed tool handlers and the loop in detail
- [Memory](/guides/memory/) to give an agent conversation history
- [Providers](/guides/providers/) to pick a model for the agent
