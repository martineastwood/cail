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

The model needs function calling support. A model without it may answer with
text instead of a call, so check `result->tool_results` before decoding one.

## Manage each round yourself

For the common case of a model with standing instructions and a fixed tool set,
`cail::Agent` runs this loop for you. See [Agent](/guides/agent/).

The lower-level `LanguageModel::generate(GenerationRequest)` interface remains
available for applications that want to run each round themselves, using the
request history and tool definitions directly.

## Stream a tool loop

Use `stream_tool_loop` when you want events from every model call while CAIL
continues to execute tools:

```cpp
std::stop_source stop;
auto result = cail::stream_tool_loop(
    cail::openai("gpt-6-luna"),
    cail::GenerationRequest{
        .messages = {cail::Message{
            .content = {cail::TextPart{.text = "Fetch the weather for Paris."}},
        }},
    },
    std::vector<cail::Tool>{weather_tool},
    [](const cail::StreamEvent& event) {
        if (const auto* delta = std::get_if<cail::TextDelta>(&event)) {
            std::cout << delta->text << std::flush;
        }
    },
    {
        .max_rounds = 4,
        .stop_when = [](const cail::GenerationResponse& response,
                        const cail::MiddlewareContext&) {
            return response.status == cail::GenerationStatus::refused;
        },
        .stop = stop.get_token(),
    });
```

The event handler receives text, reasoning, usage, and tool-call events from
each step. Call `stop.request_stop()` from another thread to cancel the active
stream.

`stop_when` runs after a model step and before its requested tools execute, so
returning `true` returns that response without starting another step.
`max_rounds` remains a final guard against unbounded tool calls.

## Next steps

- [Streaming](/guides/streaming/) for the full event list
- [Structured outputs](/guides/structured-output/) for the schema types tools use
