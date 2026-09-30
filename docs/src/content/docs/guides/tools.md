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
first tool error ends the loop, and later calls do not start.

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

## Next steps

- [Streaming](/guides/streaming/) for the full event list
- [Structured outputs](/guides/structured-output/) for the schema types tools use
