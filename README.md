# CAIL (Cpp AI Library)

A typed C++ SDK for LLM providers, with a provider-neutral model and generation API.

## Requirements

- CMake 3.31 or newer
- A C++23 compiler supported by the pinned Glaze release
- OpenSSL development files for HTTPS transport
- Network access during the initial CMake configure for pinned FetchContent dependencies

## Install as a CMake package

Install CAIL and its pinned dependencies to a prefix:

```sh
cmake -S . -B build -DCAIL_BUILD_EXAMPLES=OFF
cmake --build build
cmake --install build --prefix /path/to/cail
```

In a consuming project's `CMakeLists.txt`:

```cmake
find_package(cail 0.1 CONFIG REQUIRED)
target_link_libraries(app PRIVATE cail::cail)
```

Configure the consumer with `-DCMAKE_PREFIX_PATH=/path/to/cail`.

## Current implementation

The SDK provides typed fields, JSON conversion, generation requests, structured outputs, function tool calls, text streaming, and OpenAI, OpenRouter, Anthropic, Gemini, Mistral, Charm Hyper, and Ollama Cloud providers. It targets C++23.

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

`cail::Field<T>::value` is the runtime value; read it directly and assign through the field. JSON conversion emits only the value, while schema inspection reads the metadata. For example, `analysis.confidence.value` reads a value and `analysis.confidence = 0.8` updates it.

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

Use `LanguageModel::stream()` to receive events as they arrive. The call returns the complete `GenerationResponse` after the provider finishes:

```cpp
auto response = cail::openai("gpt-6-luna").stream("Summarize this paragraph.", [](const cail::StreamEvent& event) {
    if (const auto* delta = std::get_if<cail::TextDelta>(&event)) {
        std::cout << delta->text << std::flush;
    }
});
```

The callback can also receive `RefusalDelta`, `ReasoningDelta`, `ToolCallArgumentsDelta`, `ToolCallReady`, and `UsageUpdate`. Tool argument deltas and completed calls include an output index so you can match them. The callback runs synchronously while `stream()` processes the response. The returned result reports success or failure and includes the final text, reasoning, tool calls, and usage.

When the provider reports them, `response->usage->cache_read_tokens` and `reasoning_tokens` contain extra token counts. These fields are optional, so an absent value means the provider did not report that detail. Errors include a stable `code`; OpenAI errors can also include `http_status`, `provider_code`, `provider_type`, and `request_id`.

To cancel an active stream, pass a stop token and request a stop from another thread or from the callback:

```cpp
std::stop_source stop;
auto response = cail::openai("gpt-6-luna").stream("Write a long story.",
    [&](const cail::StreamEvent& event) {
        if (std::holds_alternative<cail::TextDelta>(event)) {
            stop.request_stop();
        }
    }, stop.get_token());
// response.error().code is cail::ErrorCode::cancelled when stopped.
```

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

With `OPENROUTER_API_KEY` set, choose a model and call it directly:

```cpp
auto response = cail::openrouter("openai/gpt-4o-mini").generate(cail::GenerationRequest{
    .messages = {cail::Message{
        .role = cail::MessageRole::user,
        .content = {cail::TextPart{.text = "Summarize this note."}},
    }},
});
```

Pass an explicit key or extra headers with `cail::create_openrouter({.api_key = key, .headers = headers})`. OpenRouter accepts text, user images, tool definitions, tool history, and typed structured output with models that support JSON Schema. It streams text, tool calls, and usage through CAIL's Chat Completions adapter. It does not accept continuation tokens.

To get a typed result with a model that supports JSON Schema, call `generate_object`:

```cpp
struct Answer { std::string language; };
auto answer = cail::generate_object<Answer>({
    .model = cail::openrouter("openai/gpt-4o-mini"),
    .prompt = "What language is CAIL written in?",
});
if (answer) std::cout << answer->language << '\n';
```

For another OpenAI-compatible endpoint, use `cail::create_chat_completions({.endpoint = url, .api_key = key})`.

Run the complete prompt, streaming, and structured output examples with your OpenRouter key:

```sh
cmake -S . -B build
cmake --build build --target cail_openrouter_prompt cail_openrouter_stream cail_openrouter_object
./build/cail_openrouter_prompt
./build/cail_openrouter_stream
./build/cail_openrouter_object
```

Set `OPENROUTER_API_KEY` in your environment before running either example.

## Use Mistral or Charm Hyper

Set `MISTRAL_API_KEY` or `HYPER_API_KEY`, then choose a model:

```cpp
auto mistral = cail::mistral("mistral-vibe-cli-with-tools");
auto hyper = cail::hyper("deepseek-v4-flash");

auto answer = cail::generate_text({
    .model = mistral,
    .prompt = "Reply with exactly OK.",
});
```

Both providers use CAIL's Chat Completions features, including text streaming and tools. Model support for images and structured output varies. Use `cail::create_mistral({.api_key = key})` or `cail::create_hyper({.api_key = key})` to pass a key or extra headers explicitly.

Run the complete prompt examples:

```sh
cmake --build build --target cail_mistral_prompt cail_hyper_prompt
./build/cail_mistral_prompt
./build/cail_hyper_prompt
```

## Use Ollama Cloud

Set `OLLAMA_API_KEY`, then select a cloud model:

```cpp
auto answer = cail::generate_text({
    .model = cail::ollama_cloud("gemma4:31b"),
    .prompt = "Reply with exactly OK.",
});
```

Ollama Cloud uses CAIL's Chat Completions features, including streaming and tools. Available features depend on the model. Use `cail::create_ollama_cloud({.api_key = key})` to pass a key or extra headers explicitly.

Run the prompt example:

```sh
cmake --build build --target cail_ollama_cloud_prompt
./build/cail_ollama_cloud_prompt
```

## Use Anthropic

With `ANTHROPIC_API_KEY` set, you can use Claude through the same generation and streaming APIs:

```cpp
auto model = cail::anthropic("claude-haiku-4-5-20251001");
auto response = cail::generate_text({
    .model = model,
    .prompt = "Explain one benefit of native C++ applications in one sentence.",
});
```

Use `cail::create_anthropic({.api_key = key, .max_tokens = 2048})` to set credentials or the response limit explicitly. You can also pass `.base_url` and `.headers`. Anthropic accepts user images, tools, and typed structured output with models that support JSON Schema. It reports reasoning and cache usage when available. This adapter rejects developer-role messages and continuation tokens.

Build and run the complete examples:

```sh
cmake --build build --target cail_anthropic_prompt cail_anthropic_stream cail_anthropic_object cail_anthropic_tool_call
./build/cail_anthropic_prompt
./build/cail_anthropic_stream
./build/cail_anthropic_object
./build/cail_anthropic_tool_call
```

## Use Gemini

With `GEMINI_API_KEY` set, you can send a prompt to a Google AI Studio model:

```cpp
auto result = cail::generate_text({
    .model = cail::gemini("gemini-3.5-flash-lite"),
    .prompt = "Reply with one word: C++.",
});
```

Use `cail::create_gemini({.api_key = key})` to pass the key explicitly. You can also set `.base_url` and `.headers`. The adapter accepts user images and tools, supports text streaming and typed structured output, and reports token usage when Google provides it. Model features and quotas vary, so check the selected model in Google AI Studio.

Run the prompt, streaming, structured output, and tool examples with:

```sh
cmake --build build --target cail_gemini_prompt cail_gemini_stream cail_gemini_object cail_gemini_tool_call
./build/cail_gemini_prompt
./build/cail_gemini_stream
./build/cail_gemini_object
./build/cail_gemini_tool_call
```

Use `model.capabilities()` to check which features the CAIL adapter can send or receive:

```cpp
if (model.capabilities().tools) {
    // You can pass tool definitions in a GenerationRequest.
}
```

The flags describe adapter support. A selected model or endpoint may support fewer features and can still reject a request. Check the provider's model documentation for model-specific availability.

The optional `system` prompt is sent as the first system message and works with both `prompt` and `messages`.

## Create embeddings

Use an embedding model when you need vectors for search or similarity. With `OPENAI_API_KEY` set, you can embed one text or a batch:

```cpp
auto model = cail::openai.embedding_model("text-embedding-3-small");
auto one = model.embed("A red apple");
auto batch = model.embed_many({"A red apple", "A green pear"});
if (batch) {
    std::cout << batch->model << ": " << batch->dimensions << " dimensions\n";
    // batch->embeddings[i] matches input i.
}
```

Each vector includes the returned model ID and dimension count. Batch results follow input order even if the provider returns indexed vectors out of order. `batch->input_tokens` is present when the provider reports usage. For OpenAI's `text-embedding-3` models, pass a positive dimension count as the second argument to `embedding_model` when you want shorter vectors. Use the same model and dimensions for vectors you compare or store together.

Build the complete example with `cmake --build build --target cail_openai_embed`, then run `./build/cail_openai_embed`.

## Send an image

You can include text and image bytes in a user message when the selected OpenAI model supports images:

```cpp
#include <cail/cail.hpp>

#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <utility>

int main()
{
    std::ifstream image("screenshot.png", std::ios::binary);
    if (!image) return 1;
    std::string bytes(std::istreambuf_iterator<char>{image}, {});

    auto response = cail::openai("gpt-6-luna").generate(cail::GenerationRequest{
        .messages = {cail::Message{
            .role = cail::MessageRole::user,
            .content = {
                cail::TextPart{.text = "What does this screenshot show?"},
                cail::ImagePart{.bytes = std::move(bytes), .mime_type = "image/png"},
            },
        }},
    });
    if (!response) {
        std::cerr << response.error().message << '\n';
        return 1;
    }
    std::cout << response->text << '\n';
}
```

Pass the original file bytes and the matching MIME type. The OpenAI Responses adapter sends images in user messages; image parts in other roles return an error.

`std::optional<T>` fields are optional in CAIL schemas and deserialize from either a missing property or `null`. OpenAI strict output and tool schemas require every property, so CAIL marks optional properties required and allows `null` in their JSON Schema type; the model returns `null` when no value is available.

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
