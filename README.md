# CAIL (Cpp AI Library)

A typed C++ SDK for LLM providers, with a provider-neutral model and generation API.

## Requirements

- CMake 3.31 or newer
- A C++23 compiler supported by the pinned Glaze release
- OpenSSL development files for HTTPS transport
- Network access during the initial CMake configure for pinned FetchContent dependencies

## Current implementation

The SDK provides typed fields, Glaze JSON conversion, provider-neutral generation requests, structured outputs, function tool calls, text streaming, and an OpenAI provider adapter. It targets C++23 and pins Glaze and `magic_enum` through CMake FetchContent.

`magic_enum` supplies enum names because Glaze's C++23 mode serializes enums as integers by default. Glaze remains the backend for struct reflection and JSON conversion.

```cpp
enum class Sentiment { Positive, Neutral, Negative };

struct Analysis {
    cail::Field<Sentiment> sentiment{
        .description = "Overall sentiment"
    };

    cail::Field<double> confidence{
        .description = "Confidence score",
        .minimum = 0.0,
        .maximum = 1.0
    };
};
```

`cail::Field<T>::value` is the explicit runtime value access for this prototype. JSON conversion emits the value only; schema inspection also reads the field metadata.

The example round-trips an `Analysis` value and emits its generated JSON Schema with `cail::json_schema<Analysis>()`.

The default OpenAI provider reads `OPENAI_API_KEY` from the environment. Pass the model into the provider-neutral generation helpers:

```cpp
auto text = cail::generate_text({
    .model = cail::openai("gpt-6-luna"),
    .system = "You are a concise assistant.",
    .prompt = "Summarize the main idea of this paragraph.",
});
auto analysis = cail::generate_object<Analysis>({
    .model = cail::openai("gpt-6-luna"),
    .prompt = "Classify this review: ...",
});
```

Use `LanguageModel::stream()` to receive text deltas as they arrive. The call returns the complete `GenerationResponse` after the provider finishes:

```cpp
auto response = cail::openai("gpt-6-luna").stream("Summarize this paragraph.", [](std::string_view delta) {
    std::cout << delta << std::flush;
});
```

The callback runs on the HTTP streaming thread. Streaming supports text and refusal deltas; the returned response also contains final usage and tool calls.

For a custom API key or base URL, create your own provider instance:

```cpp
auto openai = cail::create_openai({
    .api_key = std::getenv("OPENAI_API_KEY"),
    .base_url = "https://api.openai.com/v1",
});
auto text = cail::generate_text({
    .model = openai("gpt-6-luna"),
    .prompt = "Summarize the main idea of this paragraph.",
});
```

The optional `system` prompt is sent as the first system message and works with both `prompt` and `messages`.

OpenAI strict structured output currently requires every schema property to be required. Optional fields are rejected with an `unsupported_schema` error until nullable schema support is added.

Function tools use the same CAIL schemas. `cail::tool<Input, Output>()` creates a typed handler, and `cail::generate_text()` runs the call/execute/result loop. The returned `GenerationResponse` contains each tool result as JSON, which the tool can decode to its declared output type:

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

Tool handlers run in application code, so CAIL never guesses which local function to execute. CAIL handles dispatch, typed argument decoding, repeated model calls, and collecting the typed-decodable results. The lower-level `LanguageModel::generate(GenerationRequest)` interface remains available for applications that want to manage each round themselves.

The OpenAI prompt example makes a normal text request. Set `OPENAI_API_KEY`, then run it with:

```sh
OPENAI_API_KEY=... ./build/cail/cail_openai_prompt
```

The streaming example prints text as it arrives:

```sh
OPENAI_API_KEY=... ./build/cail/cail_openai_stream
```

The structured output example requests an `Analysis` object and prints its JSON representation:

```sh
OPENAI_API_KEY=... ./build/cail/cail_openai_generate_object
```

The OpenAI tool example demonstrates automatic tool-call looping and typed output decoding:

```sh
OPENAI_API_KEY=... ./build/cail/cail_openai_tool_call
```

Build the example with:

```sh
cmake -S . -B build/cail -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build/cail
```
