---
title: Tools
description: Let the model call typed C++ functions with CAIL's tool loop.
---

You can give a model access to your own functions. `cail::tool<Input, Output>()`
creates a typed handler, and `cail::generate_text()` runs the full
call/execute/result loop: the model requests a call, CAIL decodes the
arguments into your input type, runs your handler, and sends the result back
for the next model step.

Tool handlers run in application code. CAIL never guesses which local function
to execute; it handles dispatch, typed argument decoding, repeated model calls,
and collecting the typed-decodable results.

## Create a tool and use it

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

auto result = cail::generate_text({
    .model = cail::openai("gpt-6-luna"),
    .prompt = "Fetch the weather for Paris.",
    .tools = {weather_tool},
});
if (result && !result->tool_results.empty()) {
    auto report = weather_tool.decode_output(result->tool_results.front());
}
```

The returned `GenerationResponse` contains each tool result as JSON. Decode it
to the tool's declared output type with `decode_output`, as above.

You can constrain numeric arguments with `cail::Field<T>::minimum` and
`maximum`, strings with `min_length` and `max_length`, and vectors with
`min_items` and `max_items`. CAIL checks these inclusive bounds before calling your handler,
including asynchronous handlers. Invalid arguments return
`ErrorCode::schema_validation` with the failing field path. See
[Structured outputs](/guides/structured-output/) for a constrained field example.

The model needs function calling support. A model without it may answer with
text instead of a call, so check `result->tool_results` before decoding one.

## Let the model recover from tool errors

You can return argument and execution errors to the model so it can correct a
call or choose another approach:

```cpp
auto result = cail::generate_text({
    .model = cail::openai("gpt-6-luna"),
    .prompt = "Fetch the weather for Paris.",
    .tools = {weather_tool},
    .tool_loop = {.recover_tool_errors = true},
});
```

Recovery is off by default. When enabled, `tool_execution`, `invalid_tool_call`,
`json_deserialization`, and `schema_validation` errors become tool results with
JSON such as `{"error":"Unknown location"}`. Later calls in the batch still run,
and the model receives every result at the next step. Thrown handler exceptions
also become `tool_execution` errors.

Cancellation, missing tools, transport errors, and other error codes still end
the loop. Return `Result<Output>` from your handler with `tool_execution` for a
failure the model can act on. The existing `max_rounds` limit also bounds recovery.
Error results do not match your declared output type, so `decode_output` can fail
for these entries.

## Approve calls or execute them elsewhere

You can inspect tool calls before executing them and resume with results from
your application. This example asks for approval before fetching weather:

```cpp
#include <cail/cail.hpp>
#include <iostream>
#include <string>
#include <vector>

struct WeatherQuery { std::string location; };
struct WeatherReport { std::string location; int temperature_fahrenheit{}; };

int main() {
    auto model = cail::openai("gpt-6-luna");
    auto weather = cail::tool<WeatherQuery, WeatherReport>(
        "weather", "Fetch weather for a location.", [](const WeatherQuery& query) {
            return WeatherReport{query.location, 72}; // Replace with your weather service.
        });
    auto result = cail::generate_text({
        .model = model,
        .prompt = "Use the weather tool to fetch the weather for Paris.",
        .tools = {weather},
        .tool_loop = {.pause_when = [](const cail::ToolCall&) { return true; }},
    });
    while (result && result->tool_continuation) {
        std::vector<cail::ToolResult> outputs;
        for (const auto& call : result->tool_calls) {
            std::cout << call.name << " " << call.arguments << "\nApprove? [y/N] ";
            std::string answer;
            std::getline(std::cin, answer);
            cail::Result<std::string> output = std::string{R"({"error":"Approval denied"})"};
            if (answer == "y") output = weather.execute(call, {});
            if (!output) {
                std::cerr << output.error().message << '\n';
                return 1;
            }
            outputs.push_back({call.id, call.name, *output});
        }
        result = cail::resume_tool_loop(model, *result->tool_continuation, outputs);
    }
    if (!result) {
        std::cerr << result.error().message << '\n';
        return 1;
    }
    std::cout << result->text << '\n';
}
```

If any call matches `pause_when`, the entire batch returns before any of its
tools execute. Inspect `tool_calls` and supply exactly one `ToolResult` per call,
with matching IDs and names in the same order. You can execute calls locally,
send them to a queue, or return a JSON rejection. Tell the model in your system
prompt how it should handle rejected calls.

`tool_continuation` retains the original request options, tools, pause policy,
and round limit. Resumed responses include the earlier steps, tool results,
conversation turn, and total reported usage. A subsequent batch can pause again.

Continuations are process-local and single-use. Keep the continuation alive
until you resume it; it cannot be serialized for use after a restart. Invalid
results or a pre-cancelled stop token leave it available for correction. Once
valid results are accepted, a failed or cancelled run cannot reuse it. External
tool execution belongs to your application, so avoid executing the same call
twice while retrying a rejected resume.

For streaming, use `resume_stream_tool_loop(model, continuation, outputs, on_event)`.
For callbacks, use `resume_tool_loop_async(model, continuation, outputs, on_complete)`
or `resume_stream_tool_loop_async(model, continuation, outputs, on_event, on_complete)`.
These callback APIs accept a stop token and `AsyncOptions` after the completion
handler, including a callback scheduler and an event limit for streams.

You can also await resumption by passing the shared continuation pointer:

```cpp
auto resumed = co_await cail::resume_tool_loop_async(
    model, result->tool_continuation, outputs);
```

Use `resume_stream_tool_loop_async` with the shared pointer and an event handler
to await a resumed stream. These overloads retain the continuation until completion.

## Remember approved tool calls

You can pause an agent for approval and keep the completed exchange in conversation
memory. Use the agent's `resume` method after collecting each batch of results:

```cpp
#include <cail/cail.hpp>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

struct WeatherQuery { std::string location; };
struct WeatherReport { std::string location; int temperature_fahrenheit{}; };

int main() {
  auto weather = cail::tool<WeatherQuery, WeatherReport>(
      "weather", "Fetch weather for a location.", [](const WeatherQuery& query) {
        return WeatherReport{query.location, 72}; // Replace with your weather service.
      });
  cail::Agent agent({
      .model = cail::openai("gpt-6-luna"),
      .instructions = "Help with travel plans. If a tool result contains an error, explain it without inventing weather.",
      .tools = {weather},
      .memory = std::make_shared<cail::FileConversationMemory>("conversations"),
      .conversation_id = "traveler-42",
  });
  auto result = agent.generate("Use the weather tool to fetch the weather for Paris.",
                              {.pause_when = [](const cail::ToolCall&) { return true; }});
  while (result && result->tool_continuation) {
    std::vector<cail::ToolResult> outputs;
    for (const auto& call : result->tool_calls) {
      std::cout << call.name << " " << call.arguments << "\nApprove? [y/N] ";
      std::string answer;
      std::getline(std::cin, answer);
      cail::Result<std::string> output = std::string{R"({"error":"Approval denied"})"};
      if (answer == "y") output = weather.execute(call, {});
      if (!output) {
        std::cerr << output.error().message << '\n';
        return 1;
      }
      outputs.push_back({call.id, call.name, *output});
    }
    result = agent.resume(*result->tool_continuation, outputs);
  }
  if (!result) {
    std::cerr << result.error().message << '\n';
    return 1;
  }
  std::cout << result->text << '\n';
  auto follow_up = agent.generate("What city did I ask about?");
  if (!follow_up) {
    std::cerr << follow_up.error().message << '\n';
    return 1;
  }
  std::cout << follow_up->text << '\n';
}
```

The initial prompt, tool calls, tool results, and final answer are saved together
when the turn finishes successfully. The follow-up can recall that exchange,
including rejected calls. Build and run this example from a checkout:

```sh
cmake -S . -B build -DCAIL_BUILD_EXAMPLES=ON
cmake --build build --target cail_openai_approval_memory
export OPENAI_API_KEY="your-api-key"
./build/cail_openai_approval_memory
```

A second approval pauses the same turn again without
saving it early. Conversation ID overrides and attachments stay with the original
turn.

While approval is pending, another prompt for the same conversation returns a
`memory` error. Finish the turn or release every copy of its continuation before
starting another prompt. Discarding a continuation abandons the unfinished turn
without saving it. Calling `clear` on the memory store does not abandon approval.

Resume memory-backed continuations through the agent that started them, or a
copy of that agent. Standalone resume functions and a separately constructed agent
return `invalid_configuration`. Invalid results and pre-cancelled tokens leave
approval pending, so you can correct the results and retry. Once results are
accepted, a failed or cancelled resume consumes the continuation and releases the
conversation. Failed generation saves no partial turn. An async memory write
already in progress may finish after cancellation. Inspect `error().partial_response`
before repeating tool actions.

Use `agent.resume_stream(continuation, outputs, on_event)` to stream the next steps.
The callback forms are `agent.resume_async(continuation, outputs, on_complete)` and
`agent.resume_stream_async(continuation, outputs, on_event, on_complete)`. Pass a stop
token and `AsyncOptions` after the completion handler to cancel or schedule delivery.
To await either form, pass the shared continuation pointer:

```cpp
auto resumed = co_await agent.resume_async(result->tool_continuation, outputs);
```

Continuations remain process-local. File memory saves completed turns, so pending
approval cannot survive a restart. The agent's `GenerationRequest` overload still
uses application-managed history and saves nothing automatically. See
[Memory](/guides/memory/) for conversation setup and history limits.

## Manage each round yourself

For the common case of a model with standing instructions and a fixed tool set,
`cail::Agent` runs this loop for you. See [Agent](/guides/agent/).

The lower-level `LanguageModel::generate(GenerationRequest)` interface remains
available for applications that want to run each round themselves, using the
request history and tool definitions directly.

## Stream a tool loop

Use `stream_text` when you want events from every model call while CAIL
continues to execute tools:

```cpp
std::stop_source stop;
auto result = cail::stream_text(
    {
        .model = cail::openai("gpt-6-luna"),
        .prompt = "Fetch the weather for Paris.",
        .tools = {weather_tool},
        .tool_loop = {
            .max_rounds = 4,
            .stop_when = [](const cail::GenerationResponse& response,
                            const cail::MiddlewareContext&) {
                return response.status == cail::GenerationStatus::refused;
            },
            .stop = stop.get_token(),
        },
    },
    [](const cail::StreamEvent& event) {
        if (const auto* delta = std::get_if<cail::TextDelta>(&event)) {
            std::cout << delta->text << std::flush;
        }
    });
```

The event handler receives text, reasoning, usage, and tool-call events from
each step. Call `stop.request_stop()` from another thread to cancel the active
stream.

`stop_when` runs after a model step and before its requested tools execute, so
returning `true` returns that response without starting another step.
`max_rounds` remains a final guard against unbounded tool calls.

## Run tools without blocking generation

Use `generate_text_async` with the same tools and options as `generate_text`.
Its completion callback receives the final answer and collected tool results:

```cpp
auto started = cail::generate_text_async(
    {
        .model = cail::openai("gpt-6-luna"),
        .prompt = "Fetch the weather for Paris.",
        .tools = {weather_tool},
    },
    [](cail::Result<cail::GenerationResponse> result) {
        if (result) std::cout << result->text << '\n';
        else std::cerr << result.error().message << '\n';
    });
if (!started) {
    std::cerr << started.error().message << '\n';
}
```

Keep your application running until the callback finishes. For manually managed
requests, use `run_tool_loop_async(model, request, tools, callback, options)`.
The same `max_rounds`, `stop_when`, and middleware options apply to blocking,
streaming, and async tool loops.

Async generation runs tool handlers on a worker, so a slow synchronous tool does
not occupy the provider's completion thread. Calls execute in model order. The
first tool error ends the loop by default, and later calls do not start.
Set `recover_tool_errors` to let the model respond to argument and execution errors.

For a tool that already provides an asynchronous API, use `async_tool`. Pass the
input and context by value and call the completion callback exactly once with
`cail::Result<Output>`. You can complete inline or retain the callback until your
operation finishes:

```cpp
auto weather_tool = cail::async_tool<WeatherQuery, WeatherReport>(
    "weather", "Get the weather in a location.",
    [](WeatherQuery query, cail::ToolContext context,
       std::function<void(cail::Result<WeatherReport>)> complete) {
        if (context.stop.stop_requested()) {
            complete(std::unexpected(cail::generation_cancelled_error()));
            return;
        }
        complete(WeatherReport{.location = query.location,
                               .temperature_fahrenheit = 72});
    });
```

Set `tool_loop.stop` to a token from `std::stop_source`. Cancellation prevents
subsequent tools and model steps from starting. While waiting for an async tool,
the loop returns a cancellation error without waiting for that tool's callback.
A late callback is safely ignored. Your tool still owns its active work and must
observe `context.stop` to stop it. A synchronous handler must return before the
loop can finish cancellation. Waiting for an async tool's callback does not
occupy a CAIL worker.

You can also use an async tool with blocking generation, which waits for its
completion. Keep your application running until generation and any cancelled
tool work have finished. Handle errors in your completion callback. CAIL contains
exceptions thrown by terminal completion callbacks and does not call them again.
Callbacks that access application state must synchronize access from other threads.

## Stream without blocking the caller

`stream_text_async` returns after accepting the stream. Its completion callback
receives the final response after all tool rounds:

```cpp
std::stop_source stop;
auto started = cail::stream_text_async(
    {.model = cail::openai("gpt-6-luna"),
     .prompt = "Fetch the weather for Paris.",
     .tools = {weather_tool},
     .tool_loop = {.stop = stop.get_token()}},
    [](const cail::StreamEvent& event) {
        if (const auto* delta = std::get_if<cail::TextDelta>(&event))
            std::cout << delta->text << std::flush;
    },
    [](cail::Result<cail::GenerationResponse> result) {
        if (!result) std::cerr << result.error().message << '\n';
    });
if (!started) std::cerr << started.error().message << '\n';
```

Events arrive serially on the provider's delivery thread. Each handler must
return before the stream continues reading, so slow consumers apply backpressure.
Built-in HTTP providers deliver events on an I/O thread. Start async requests
from an event handler; a blocking HTTP request there returns an
`invalid_configuration` error.

Built-in providers use native async streaming, so active streams do not occupy
a blocking worker. Up to four synchronous tool handlers can run concurrently,
with up to 256 queued jobs. A full queue returns an
`invalid_configuration` error. For a tool-loop step that cannot be queued, the
error reaches the generation completion callback. Cancelling queued work
completes it without starting its handler or waiting for an available worker.

Call `stop.request_stop()` to cancel. No further events are delivered once the
provider observes cancellation. An active event handler must return before
cancellation can finish. An event handler exception ends the stream with an
error. An initiation error is returned directly without calling completion;
a successfully started operation calls completion once.

For manually managed requests, use `LanguageModel::stream_async` or
`stream_tool_loop_async`, or `agent.stream_async(input, on_event, complete, options)`. The same `max_rounds`, `stop_when`, and middleware
options apply.

## Inspect a failed run

You can inspect completed work when a tool loop fails:

```cpp
auto response = cail::generate_text({
    .model = cail::openai("gpt-6-luna"),
    .prompt = "Fetch the weather for Paris.",
    .tools = {weather_tool},
});
if (!response) {
    std::cerr << response.error().message << '\n';
    if (const auto& partial = response.error().partial_response) {
        for (const auto& tool : partial->tool_results)
            std::cout << tool.name << ": " << tool.output << '\n';
    }
}
```

`partial_response` includes completed model `steps`, reported `total_usage`,
successful `tool_results`, and the messages in `turn`. It is absent when no model
response was received. Blocking, streaming, callback, and coroutine tool loops
provide the same progress information.

A failed run can have already performed tool actions. Inspect its results before
retrying. Partial turns can contain unanswered tool calls, so resolve those calls
before using the messages as conversation history. Failed agent calls do not save
partial turns to memory automatically. Progress does not include unfinished
streamed responses or a resumable checkpoint.


## Next steps

- [Streaming](/guides/streaming/) for the full event list
- [Structured outputs](/guides/structured-output/) for the schema types tools use
