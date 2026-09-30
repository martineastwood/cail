---
title: Async and coroutines
description: Await generation, agent conversations, streams, and embeddings with C++ coroutines.
---

You can write async CAIL calls in sequence with `co_await`. Generation, tools,
memory, and HTTP requests finish before your next statement runs, while your
coroutine yields to other work during the wait.

## Start with an agent

Include `<cail/cail.hpp>`, set `OPENAI_API_KEY`, and write your workflow as a
`cail::Task`. Use `cail::run` to run it from `main`:

```cpp
#include <cail/cail.hpp>

#include <iostream>

cail::Task<int> chat() {
    cail::Agent agent({.model = cail::openai("gpt-6-luna")});
    auto response = co_await agent.generate_async("Explain RAII in one sentence.");
    if (!response) {
        std::cerr << response.error().message << '\n';
        co_return 1;
    }
    std::cout << response->text << '\n';
    co_return 0;
}

int main() { return cail::run(chat()); }
```

Build and link this example with the same `cail::cail` target as a synchronous
program. You do not need an additional dependency or build option. In a CAIL
source checkout, you can run the supplied example:

```sh
cmake -S . -B build -DCAIL_BUILD_EXAMPLES=ON
cmake --build build --target cail_async
export OPENAI_API_KEY="your-api-key"
./build/cail_async
```

Omitting the completion callback selects the coroutine overload of
`generate_async`. Awaiting it returns `Result<GenerationResponse>`. Both request
validation errors and errors that occur during generation arrive in that result,
so you need only one error check.

`cail::Task<T>` describes a coroutine that returns `T`. Use `co_return` to return
its value. `cail::run` waits for your workflow and returns that value to `main`.
It blocks the calling thread, so use it at a synchronous entry point. Avoid
calling it from an I/O callback. Inside an
async workflow, use `co_await` instead of calling `run` again. Exceptions thrown
by your own workflow propagate out of `run`.

The remaining examples belong inside a `cail::Task` function.

## Generate text or a typed object

The free functions work the same way:

```cpp
auto response = co_await cail::generate_text_async({
    .model = cail::openai("gpt-6-luna"),
    .prompt = "Suggest a name for a neighborhood bakery.",
});
```

For typed output, use your usual CAIL output type:

```cpp
struct Bakery {
    std::string name;
    std::string tagline;
};

auto bakery = co_await cail::generate_object_async<Bakery>({
    .model = cail::openai("gpt-6-luna"),
    .prompt = "Suggest a name and tagline for a neighborhood bakery.",
});
if (bakery) std::cout << bakery->name << '\n';
```

Tools and generation options work as they do with the callback APIs. An awaited
agent call finishes its tool rounds and saves a successful conversation turn
before returning. User messages use memory when configured; explicit
`GenerationRequest` values bypass it.

## Stream text and await the final response

Pass an event handler and await the result. You do not need a completion handler:

```cpp
auto response = co_await cail::stream_text_async(
    {.model = cail::openai("gpt-6-luna"),
     .prompt = "Write a short welcome message for a bakery."},
    [](const cail::StreamEvent& event) {
        if (const auto* text = std::get_if<cail::TextDelta>(&event))
            std::cout << text->text << std::flush;
    });
if (!response) std::cerr << response.error().message << '\n';
```

You can also use `co_await agent.stream_async(prompt, on_event)` with a prompt,
user message, or explicit request.

By default, events run serially on your coroutine's executor. The awaited final
result follows event delivery. If a consumer falls behind, the event queue is
bounded by `max_pending_events`, which defaults to 256. A full queue cancels the
stream with `ErrorCode::backpressure`. See [Streaming](/guides/streaming/) for
queue limits and callback scheduling options.

## Await embeddings

Single and batch embedding calls return their existing result types:

```cpp
auto model = cail::openai.embedding_model("text-embedding-3-small");
auto embedding = co_await model.embed_async("Fresh bread every morning.");
auto batch = co_await model.embed_many_async(
    std::vector<std::string>{"Sourdough", "Croissants"});
```

## Cancel an operation

Pass a stop token through the usual options:

```cpp
std::stop_source stop;
auto response = co_await agent.generate_async(
    "Plan a bakery opening.",
    cail::ToolLoopOptions{.stop = stop.get_token()});
```

Another part of your application can call `stop.request_stop()`. Cancellation
returns an error with `ErrorCode::cancelled`. The token applies to memory work,
HTTP requests, retries, and tools. A blocking tool or memory write already running
may need to finish before cancellation completes.

## Use your existing Asio event loop

`cail::Task` works with the Asio backend selected by CAIL's build. You can await
CAIL calls from an existing compatible Asio coroutine, or spawn a workflow on
your application's executor:

```cpp
cail::asio::io_context context;
auto result = cail::asio::co_spawn(context, chat(), cail::asio::use_future);
context.run();
int exit_code = result.get();
```

Coroutine continuations return to the executor where the task is running, even
when a provider completes on another thread. Asio cancellation signals propagate
to active CAIL operations when you spawn a task with a cancellation slot. The
callback APIs remain available
when you need to integrate with a different scheduler or framework.

## Task ownership

Creating a task does not start its request. Await it or spawn it to start the
operation. Discarding a task before it starts sends no request. Stored tasks are
move-only, so move a stored task when awaiting it:

```cpp
auto task = agent.generate_async("Suggest tomorrow's specials.");
auto response = co_await std::move(task);
```

CAIL tasks own their model, agent, and input values. Keep any state captured by
reference in tools, middleware, or event handlers alive until the awaited call
finishes. Keep your event loop running until active workflows finish.

## Next steps

- [Structured outputs](/guides/structured-output/) for `co_await generate_object_async`
- [Request controls and results](/guides/request-controls/) for sampling and usage on async calls
- [Memory](/guides/memory/) to keep a conversation across awaited calls
- [Tools](/guides/tools/) to let an agent call your application functions
- [Advanced usage](/guides/advanced/) for callback-based integrations and retries
