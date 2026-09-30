# CAIL (Cpp AI Library)

A typed C++ SDK for LLM providers, with a provider-neutral model and generation API.

> **CAIL is in alpha.** Interfaces can change without notice between releases.
> Pin the version you consume and check the API before you upgrade.

The documentation lives at <https://martineastwood.github.io/cail/> and is built from the
files in [`docs/`](docs/). Run the site locally with `npm run dev` in `docs/`.

## Your first request

Set an API key in your environment and send a prompt:

```cpp
auto response = cail::generate_text({
    .model = cail::openai("gpt-6-luna"),
    .system = "You are a concise assistant.",
    .prompt = "Summarize the main idea of this paragraph.",
});
```

The optional `system` prompt is sent as the first system message and works with both
`prompt` and `messages`.

## What you can do

You write against one provider-neutral generation API and pass in the provider. Today CAIL
supports:

- OpenAI, OpenRouter, Azure Foundry, Anthropic, Gemini, Mistral, Charm Hyper, Ollama Cloud,
  OpenCode, and local OpenAI-compatible servers
- Text streaming with typed events, including reasoning and tool-call deltas
- Structured outputs decoded into your C++ types
- Function tools with typed argument decoding and an automatic tool loop
- Agents that bundle a model, standing instructions, and tools
- Embeddings from any provider
- An in-memory embedding store for search over small collections
- Local file, text, image, and PDF loaders
- Image and PDF inputs where the provider supports them

CAIL is header-only and targets C++23.

## Install

You need CMake 3.31 or newer, a C++23 compiler supported by the pinned Glaze release, and
OpenSSL development files for HTTPS transport.

CAIL builds against [Glaze](https://github.com/stephenberry/glaze) 8.4.0 or newer and
[magic_enum](https://github.com/Neargye/magic_enum). The first CMake configure downloads
them unless you already have them installed. Install them yourself, or point CMake at a
prefix that contains them, to build without network access:

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/deps
```

### Install as a CMake package

Install CAIL and its pinned dependencies to a prefix:

```sh
cmake -S . -B build -DCAIL_BUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF
cmake --build build
cmake --install build --prefix /path/to/cail
```

In a consuming project's `CMakeLists.txt`:

```cmake
find_package(cail 0.1 CONFIG REQUIRED)
target_link_libraries(app PRIVATE cail::cail)
```

Configure the consumer with `-DCMAKE_PREFIX_PATH=/path/to/cail`.

Installing CAIL also installs Glaze and magic_enum into the same prefix, so a project that
consumes the installed package resolves both through `CMAKE_PREFIX_PATH` alone.

### Build and test from source

The `./dev` script wraps configure, build, test, formatting, and static analysis. Run
everything the way CI does:

```sh
./dev check
```

Run one suite while you iterate on an adapter:

```sh
./dev configure
./dev build build/dev cail_openai_test
./dev test build/dev cail_openai
```

Other unit-test groups are `cail_core`, `cail_chat_completions`, `cail_foundry`, and
`cail_opencode`. CAIL also provides `cail_local_http`, which starts a local Python server,
and `cail_install_smoke`, which builds a consumer against the installed package.

Run the remaining checks before you open a pull request:

```sh
./dev format        # Check formatting. ./dev format --fix rewrites the files.
./dev tidy          # clang-tidy over the test translation units.
./dev sanitizer     # Build and test with ASan and UBSan.
```

Formatting is pinned to `clang-format` 23.1.1. `./dev format` warns when your local
version differs from the one CI uses.

## Load a local file

You can attach a local PDF to a user message:

```cpp
auto pdf = cail::load_pdf("report.pdf");
if (!pdf) {
    std::cerr << pdf.error().message << '\n';
    return 1;
}
auto response = cail::generate_text({
    .model = cail::openai("gpt-6-luna"),
    .messages = {cail::Message{
        .content = {cail::TextPart{.text = "Summarize this report."}, std::move(*pdf)},
    }},
});
```

Include `<cail/loaders.hpp>` or `<cail/cail.hpp>`. Use `load_text` for text,
`load_image` for PNG, JPEG, GIF, or WebP, and `load_file` for raw bytes. All loaders
return `Result<T>` and limit source files to 32 MiB by default.

PDF inputs require a supporting model through OpenAI Responses, Anthropic, or
Gemini. See [Files, images, and PDFs](docs/src/content/docs/guides/loaders.md)
for complete examples, size limits, and error handling.

## Structured outputs

Describe a result with `cail::Field<T>` members and call `cail::generate_object<T>()`:

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

auto analysis = cail::generate_object<Analysis>({
    .model = cail::openai("gpt-6-luna"),
    .prompt = "Classify this review: ...",
});
```

`cail::Field<T>::value` is the runtime value; read it directly and assign through the
field. JSON conversion emits only the value, while schema inspection reads the metadata.
For example, `analysis.confidence.value` reads a value and `analysis.confidence = 0.8`
updates it. Call `cail::json_schema<Analysis>()` when you want the generated JSON Schema.

When defining a schema directly, set `Schema::min_items` and `Schema::max_items` to limit
array lengths. `Schema::items` describes each array value.

`std::optional<T>` fields are optional in CAIL schemas and deserialize from either a
missing property or `null`. OpenAI strict output and tool schemas require every property,
so CAIL marks optional properties required and allows `null` in their JSON Schema type;
the model returns `null` when no value is available.

The model needs JSON Schema support for structured output. OpenAI, Anthropic, Gemini,
OpenRouter with supporting models, and Chat Completions providers that accept JSON Schema
all work.

## Tools

`cail::tool<Input, Output>()` creates a typed handler, and `cail::generate_text()` runs
the call/execute/result loop. The returned `GenerationResponse` contains each tool result
as JSON, which the tool can decode to its declared output type:

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

Tool handlers run in application code, so CAIL never guesses which local function to
execute. CAIL handles dispatch, typed argument decoding, repeated model calls, and
collecting the typed-decodable results.

Stream a tool loop when you want events from every model call while CAIL continues to
execute tools:

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

The event handler receives text, reasoning, usage, and tool-call events from each step.
Call `stop.request_stop()` from another thread to cancel the active stream. `stop_when`
runs after a model step and before its requested tools execute, so returning `true`
returns that response without starting another step. `max_rounds` remains a final guard
against unbounded tool calls.

## Agents

Use `Agent` when you want to reuse the same model, instructions, and tools across prompts:

```cpp
cail::Agent weather_agent({
    .model = cail::openai("gpt-6-luna"),
    .instructions = "You help travelers plan around the weather.",
    .tools = {weather_tool},
});

auto result = weather_agent.generate("What should I pack for Paris?");
```

An agent runs the same bounded tool loop as `generate_text`. It sends its instructions as
the first system message, executes tool calls, returns their results to the model, and
stops when the model answers or the round limit is reached.

For a conversation that should remember earlier messages, attach the file-backed memory
backend:

```cpp
cail::Agent weather_agent({
    .model = cail::openai("gpt-6-luna"),
    .instructions = "You help travelers plan around the weather.",
    .tools = {weather_tool},
    .memory = std::make_shared<cail::FileConversationMemory>("conversations"),
    .conversation_id = "traveler-42",
});

auto first = weather_agent.generate("What should I pack for Paris?");
auto second = weather_agent.generate("What about shoes?");
```

`FileConversationMemory` stores one JSON file per conversation id, so the second prompt can
use the first prompt and response, including any tool calls and results. The history is
loaded before each prompt and appended after a successful call. Set `keep_last_messages`
on the options to limit how much history is sent.

If your history lives in a database or another store, implement the
`cail::ConversationMemory` interface and pass it as `.memory` instead. It provides
`load`, `append`, and `clear` operations for each conversation id. See
[Memory](https://martineastwood.github.io/cail/guides/memory/) for the built-in and custom
backends.

Pass a single user `Message` to remember images and PDFs alongside text:

```cpp
auto pdf = cail::load_pdf("report.pdf");
if (!pdf) {
    std::cerr << pdf.error().message << '\n';
    return 1;
}
auto first = weather_agent.generate(cail::Message{
    .content = {cail::TextPart{.text = "Summarize this report."}, std::move(*pdf)},
});
if (!first) {
    std::cerr << first.error().message << '\n';
    return 1;
}
auto follow_up = weather_agent.generate("What is the main conclusion?");
```

`generate`, `stream`, and `generate_async` accept user messages through the same
memory path. File memory preserves attachment bytes across restarts. Existing
version 1 conversation files must be cleared before reuse with the new version 2
format. See [Memory](docs/src/content/docs/guides/memory.md) for supported inputs,
history limits, and resetting older conversations.

For full control, pass a `GenerationRequest`. This overload bypasses memory: CAIL sends
the messages you provide and does not load or store anything. It is useful for manually
managed history and per-request options:

```cpp
std::vector<cail::Message> history = {
    cail::Message{
        .content = {cail::TextPart{.text = "What should I pack for Paris?"}},
    },
    cail::Message{
        .role = cail::MessageRole::assistant,
        .content = {cail::TextPart{
            .text = "Bring a light rain jacket and comfortable layers.",
        }},
    },
    cail::Message{
        .content = {cail::TextPart{.text = "What about shoes?"}},
    },
};

auto result = weather_agent.generate(cail::GenerationRequest{
    .messages = history,
    .max_output_tokens = 256,
});
if (result) {
    history.insert(history.end(), result->turn.begin(), result->turn.end());
}
```

`history` must include the current user message because a `GenerationRequest` has no
separate prompt field. Your application owns this vector and should append
`result->turn` after successful calls. Keep the agent's instructions out of `history`,
because `Agent` adds them as the system message.

## Streaming

Use `stream_text()` with the same options as `generate_text()` to receive events as they arrive. The call returns the
complete `GenerationResponse` after the provider finishes:

```cpp
auto response = cail::stream_text(
    {.model = cail::openai("gpt-6-luna"), .prompt = "Summarize this paragraph."},
    [](const cail::StreamEvent& event) {
        if (const auto* delta = std::get_if<cail::TextDelta>(&event)) {
            std::cout << delta->text << std::flush;
        }
    });
```

The callback can also receive `RefusalDelta`, `ReasoningDelta`, `ToolCallArgumentsDelta`,
`ToolCallReady`, and `UsageUpdate`. Tool argument deltas and completed calls include an
output index so you can match them. The callback runs synchronously while `stream_text()`
processes the response. The returned result reports success or failure and includes the
final text, reasoning, tool calls, and usage.

When the provider reports them, `response->usage->cache_read_tokens` and
`reasoning_tokens` contain extra token counts. These fields are optional, so an absent
value means the provider did not report that detail.

Chat Completions streaming includes usage by default. Set `stream_usage` to `false` when
your endpoint does not support `stream_options.include_usage`.

To cancel an active stream, pass a stop token and request a stop from another thread or
from the callback:

```cpp
std::stop_source stop;
auto response = cail::stream_text(
    {
        .model = cail::openai("gpt-6-luna"),
        .prompt = "Write a long story.",
        .tool_loop = {.stop = stop.get_token()},
    },
    [&](const cail::StreamEvent& event) {
        if (std::holds_alternative<cail::TextDelta>(event)) {
            stop.request_stop();
        }
    });
// response.error().code is cail::ErrorCode::cancelled when stopped.
```

## Send an image

You can include text and image bytes in a user message when the selected OpenAI model
supports images:

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

Pass the original file bytes and the matching MIME type. Image parts are supported in user
and tool-result messages; assistant messages cannot contain images. If you build
tool-result history yourself, add an `ImagePart` to the tool message content and CAIL will
encode it for the selected provider.

## Create embeddings

Use an embedding model when you need vectors for search or similarity. Build one from any
provider, then embed one text or a batch:

```cpp
auto model = cail::openai.embedding_model("text-embedding-3-small");
auto one = model.embed("A red apple");
auto batch = model.embed_many({"A red apple", "A green pear"});
if (batch) {
    std::cout << batch->model << ": " << batch->dimensions << " dimensions\n";
    // batch->embeddings[i] matches input i.
}
```

Every provider exposes an `embedding_model`. Pass the embedding model ID your provider
documents, the same way you pass a chat model ID to the provider itself. CAIL ships no
model catalog, so the lookup stays with you.

The second argument is optional. Pass a positive dimension count when you want shorter
vectors from a model that supports them:

```cpp
auto model = cail::openai.embedding_model("text-embedding-3-small", 512);
```

Use the same model and dimensions for vectors you compare or store together.

Each vector includes the returned model ID and dimension count. Batch results follow
input order even if the provider returns indexed vectors out of order.
`batch->input_tokens` is present when the provider reports usage.

Embeddings follow the base URL you configured for the provider, so pointing a provider at
a gateway or proxy moves both chat and embeddings. Servers that need no API key, such as
`cail::local`, send no `Authorization` header. Azure Foundry is configured per deployment,
so pass the embeddings URL explicitly:

```cpp
auto model = cail::foundry.embedding_model("text-embedding-3-small", embeddings_url);
```

### Search a small collection

Use `cail::EmbeddingStore` when you want to search a handful of documents by meaning
without a database. Include `<cail/embedding_store.hpp>`, add documents, then search with
a query string:

```cpp
#include <cail/embedding_store.hpp>

cail::EmbeddingStore store(cail::openai.embedding_model("text-embedding-3-small"));
store.add({
    {.id = "apple", .text = "A red apple"},
    {.id = "pear", .text = "A green pear"},
});

auto results = store.search("A crunchy fruit", 2);
if (results) {
    for (const auto& result : *results) {
        std::cout << result.document.id << ' ' << result.score << '\n';
    }
}
```

`add` embeds the documents in one batch. Adding a document with an existing `id` replaces
the stored text and vector. Call `clear()` to remove every document and start over.
`search` embeds the query and returns up to `top_k` documents (four by default), best match
first, scored by cosine similarity. Both return an error rather than throwing. The store
keeps everything in memory and compares each query against every document, so reach for a
dedicated vector database when your collection grows.

## Providers

Most providers read their API key from a known environment variable, and each has a
`create_*` function for passing a key, base URL, or extra headers explicitly.

### OpenAI

`cail::openai` reads `OPENAI_API_KEY` from the environment. For a custom API key or base
URL, create your own provider instance:

```cpp
auto openai = cail::create_openai({
    .api_key = std::getenv("OPENAI_API_KEY"),
    .base_url = "https://api.openai.com/v1",
});
auto response = cail::generate_text({
    .model = openai("gpt-6-luna"),
    .prompt = "Summarize the main idea of this paragraph.",
});
```

### OpenRouter

With `OPENROUTER_API_KEY` set, choose a model and call it directly:

```cpp
auto response = cail::openrouter("openai/gpt-4o-mini").generate(cail::GenerationRequest{
    .messages = {cail::Message{
        .role = cail::MessageRole::user,
        .content = {cail::TextPart{.text = "Summarize this note."}},
    }},
});
```

Pass an explicit key or extra headers with
`cail::create_openrouter({.api_key = key, .headers = headers})`. OpenRouter accepts text,
user images, tool definitions, tool history, and typed structured output with models that
support JSON Schema. It streams text, tool calls, and usage through CAIL's Chat
Completions adapter. It does not accept continuation tokens.

### Anthropic

With `ANTHROPIC_API_KEY` set, you can use Claude through the same generation and streaming
APIs:

```cpp
auto model = cail::anthropic("claude-haiku-4-5-20251001");
auto response = cail::generate_text({
    .model = model,
    .prompt = "Explain one benefit of native C++ applications in one sentence.",
});
```

Use `cail::create_anthropic({.api_key = key, .max_tokens = 2048})` to set credentials or
the response limit explicitly. You can also pass `.base_url` and `.headers`. Anthropic
accepts user images, tools, and typed structured output with models that support JSON
Schema. It reports reasoning and cache usage when available. This adapter rejects
developer-role messages and continuation tokens.

### Gemini

With `GEMINI_API_KEY` set, you can send a prompt to a Google AI Studio model:

```cpp
auto result = cail::generate_text({
    .model = cail::gemini("gemini-3.5-flash-lite"),
    .prompt = "Reply with one word: C++.",
});
```

Use `cail::create_gemini({.api_key = key})` to pass the key explicitly. You can also pass
`.base_url` and `.headers`. The adapter accepts user images and tools, supports text
streaming and typed structured output, and reports token usage when Google provides it.
Model features and quotas vary, so check the selected model in Google AI Studio.

### Mistral and Charm Hyper

Set `MISTRAL_API_KEY` or `HYPER_API_KEY`, then choose a model:

```cpp
auto mistral = cail::mistral("mistral-vibe-cli-with-tools");
auto hyper = cail::hyper("deepseek-v4-flash");
```

Both providers use CAIL's Chat Completions features, including text streaming and tools.
Model support for images and structured output varies. Use
`cail::create_mistral({.api_key = key})` or `cail::create_hyper({.api_key = key})` to pass
a key or extra headers explicitly.

### Azure Foundry

Azure Foundry serves each model through its OpenAI Responses-compatible endpoint. Set
`AZURE_FOUNDRY_API_KEY`, then pass the deployment's Responses endpoint URL and deployment
name together:

```cpp
auto model = cail::create_foundry_model({
    .endpoint = "https://<resource>.cognitiveservices.azure.com/openai/responses"
                 "?api-version=2025-04-01-preview",
    .deployment = "my-deployment",
});
```

Copy the endpoint URL and API version shown for your deployment in the Azure portal. The
provider sends the deployment name as the model and supports streaming, tools, images, and
typed structured output the same way the OpenAI provider does. Pass `.api_key` explicitly
when you need to set the key other than through the environment.

### OpenCode

OpenCode models use different request formats. Choose the API family when you select a
model, and pass a stable session ID from your application for each conversation:

```cpp
auto opencode = cail::create_opencode({
    .service = cail::OpenCodeService::zen,
});

auto response = cail::generate_text({
    .model = opencode("gpt-6-luna", cail::OpenCodeApiFamily::responses),
    .prompt = "Explain one benefit of native C++ applications in one sentence.",
    .session_id = "conversation-42",
});
```

Set `OPENCODE_API_KEY` before running, or pass `.api_key` to `create_opencode`. Choose
`OpenCodeService::zen` or `OpenCodeService::go` for the service. CAIL supports the
`chat_completions`, `responses`, `anthropic_messages`, and `gemini` API families.
Availability depends on the service and model you select. OpenCode's SystemOne decision
endpoint uses a separate request format and is not available through this text generation
provider.

Every OpenCode request includes the value from `session_id` in `x-opencode-session`. Reuse
the same ID for requests in one conversation, and choose the ID in your application.
Requests with an empty session ID return an `invalid_configuration` error. `generate_text`
and tool follow-up requests preserve the session ID; you can also set it on a
`GenerationRequest` directly.

### Ollama Cloud

Set `OLLAMA_API_KEY`, then select a cloud model:

```cpp
auto answer = cail::generate_text({
    .model = cail::ollama_cloud("gemma4:31b"),
    .prompt = "Reply with exactly OK.",
});
```

Ollama Cloud uses CAIL's Chat Completions features, including streaming and tools.
Available features depend on the model. Use `cail::create_ollama_cloud({.api_key = key})`
to pass a key or extra headers explicitly.

### Local models

You can run a model on your own machine and use it through the same generation and
streaming APIs. `cail::local` points at `http://127.0.0.1:8080/v1/chat/completions`, the
default endpoint of `llama-server`:

```cpp
auto answer = cail::generate_text({
    .model = cail::local("qwen3-1.7b"),
    .prompt = "Reply with exactly OK.",
});
```

The local provider uses CAIL's Chat Completions features, so streaming, tools, and typed
structured output work the same way as with the hosted Chat Completions providers. A local
server usually needs no API key, so CAIL sends no `Authorization` header unless you set
one.

Pass the endpoint for your runtime with `cail::create_local`. Ollama serves the same API
on its own port:

```cpp
auto model = cail::create_local({.endpoint = "http://127.0.0.1:11434/v1/chat/completions"})(
    "qwen3-1.7b-local:latest");
```

Use `.endpoint` for any OpenAI-compatible server, such as LM Studio, vLLM, or a
`llama-server` started on another host or port. Set `.api_key` when your server requires
one, and `.headers` for extra request headers. The model string is the model name your
server reports, which for llama.cpp is the name in its startup output and for Ollama is
the name from `ollama list`.

Local servers vary in what they implement, so check your runtime for streaming, tools, and
JSON Schema support. A model without function calling support may answer with text instead
of a call, so check `result->tool_results` before decoding one. CAIL reports an error when
the server rejects a request.

### Any OpenAI-compatible endpoint

For another Chat Completions endpoint, use
`cail::create_chat_completions({.endpoint = url, .api_key = key})`. Set
`.request_session_header` when the endpoint routes requests by a session ID; CAIL sends
the value from `GenerationRequest::session_id` in that header.

### Check adapter capabilities

Use `model.adapter_capabilities()` to check what the CAIL integration for that API can
encode, decode, and stream:

```cpp
if (model.adapter_capabilities().tools) {
    // This adapter can attach tool definitions to the request.
}
```

These flags describe adapter support, not model support. For example,
`mistral.adapter_capabilities().image_input == true` means CAIL can pass image parts
through the Mistral Chat Completions integration. It does not mean every Mistral model
accepts images.

If you send a feature the adapter supports but the selected model does not, the provider
can still reject the request. Check the provider's model documentation for model-specific
availability.

## Control execution

`generate()`, `stream()`, embeddings, and tool loops block the calling thread until they
finish. You can schedule these calls on your application's executor and share a provider
model across concurrent tasks.

For example, you can move a request off the current thread with `std::async`:

```cpp
auto model = cail::openai("gpt-6-luna");
auto pending = std::async(std::launch::async, [&] {
    return cail::generate_text({.model = model, .prompt = "Summarize this document."});
});

do_other_work();
auto response = pending.get();
```

The default HTTP transport supports concurrent calls. A custom transport must also support
calls from multiple threads. Pass a `std::stop_token` to `model.generate(request, token)`
to cancel a nonstreaming request.

All built-in provider families support nonblocking generation. Use
`generate_text_async(options, callback)` for text and tool loops,
`generate_object_async<T>(options, callback)` for typed output, or
`agent.generate_async(input, callback, options)` for agents. Set `tool_loop.stop`
on generation options to cancel, or `stop` on agent options.

For manually managed requests, use `model.generate_async(request, callback, token)`.
Check `model.adapter_capabilities().async_generation` when using a custom model.
Embedding models provide `embed_async(text, callback, token)` and
`embed_many_async(texts, callback, token)`, with `supports_async()` for custom models.
Blocking embedding calls also accept a stop token.

Async callbacks and tool handlers may run on a CAIL I/O thread. Tool handlers
receive `ToolContext::stop` for cooperative cancellation. Keep reference captures
valid until completion. See [Advanced usage](docs/src/content/docs/guides/advanced.md)
for a complete example and callback behavior.

Provider requests time out after 30 seconds and retry 429 and 5xx responses twice with
exponential backoff. You can tune both when you create a model:

```cpp
auto model = cail::openai(
    "gpt-6-luna",
    cail::make_default_http_transport({
        .timeout = std::chrono::seconds{60},
        .retry = {.max_retries = 4},
    }));
```

Set `max_retries` to `0` to disable retries. Streaming requests retry only when no
response data has reached your callback, so a retry never duplicates an emitted chunk.

Set `max_output_tokens` to cap a generation. The limit also applies to tool follow-up
requests:

```cpp
auto response = cail::generate_text({
    .model = cail::openai("gpt-6-luna"),
    .prompt = "Summarize this paragraph in three sentences.",
    .max_output_tokens = 256,
});
```

Add middleware when you need headers, logging, metrics, or tracing around generation:

```cpp
cail::GenerationMiddleware tracing{
    .before_request = [](cail::HttpRequest& request, const cail::MiddlewareContext& context) {
        request.headers.push_back({.name = "x-model-step", .value = std::to_string(context.step)});
        std::println("starting model step {}", context.step);
    },
    .after_response = [](const cail::HttpResponse& response,
                         const cail::MiddlewareContext& context) {
        std::println("step {} returned HTTP {}", context.step, response.status_code);
    },
    .after_step = [](const cail::Result<cail::GenerationResponse>& result,
                     const cail::MiddlewareContext& context) {
        std::println("step {} {}", context.step, result ? "completed" : "failed");
    },
};

auto response = cail::generate_text({
    .model = cail::openai("gpt-6-luna"),
    .prompt = "What is the capital of France?",
    .middleware = {tracing},
});
```

Middleware runs in registration order. `before_request` can mutate the encoded HTTP
request, `after_response` observes the raw response, and `after_step` receives the decoded
result. A tool loop increments `context.step` for every follow-up model call. You can use
these hooks with your OpenTelemetry SDK to inject trace headers and record HTTP status,
errors, token usage, and tool-loop steps. CAIL does not require an OpenTelemetry
dependency.

You can pass provider-specific JSON fields on a request or in message history when a
provider needs data beyond the common API:

```cpp
auto response = cail::generate_text({
    .model = cail::openai("gpt-6-luna"),
    .prompt = "Solve this problem carefully.",
    .provider_options = {{"reasoning_effort", "high"}},
});
```

`ProviderOptions` is a JSON object, so values can be strings, numbers, booleans, arrays,
nested objects, or `null`. Use only fields accepted by the provider you selected.

CAIL returns provider metadata in `GenerationResponse::provider_options`. You can save it
with the assistant message and send it back on a follow-up request:

```cpp
auto model = cail::openrouter("openai/gpt-4o-mini");
cail::GenerationRequest request{
    .messages = {cail::Message{
        .role = cail::MessageRole::user,
        .content = {cail::TextPart{.text = "What is CAIL?"}},
    }},
};
auto response = model.generate(request);
if (response) {
    request.messages.push_back(cail::Message{
        .role = cail::MessageRole::assistant,
        .content = {cail::TextPart{.text = response->text}},
        .provider_options = response->provider_options,
    });
    request.messages.push_back(cail::Message{
        .role = cail::MessageRole::user,
        .content = {cail::TextPart{.text = "Tell me one more thing."}},
    });
    auto follow_up = model.generate(request);
}
```

For Chat Completions providers, request options are added to the request body. Message,
content-part, and tool-definition options are added to their matching history entries.

Errors include a stable `code`; OpenAI errors can also include `http_status`,
`provider_code`, `provider_type`, and `request_id`.

## Compile time

CAIL is header-only, so each translation unit that includes it also parses Glaze, Asio, and
OpenSSL. Measured on an Apple M1 Pro with AppleClang 17, best of three runs:

| Included header | `-O0` | `-O2` |
| --- | --- | --- |
| `<cail/error.hpp>` | 0.1 s | 0.1 s |
| `<cail/generation.hpp>` | 1.2 s | 1.2 s |
| `<cail/openai.hpp>` | 6.7 s | 6.8 s |
| `<cail/cail.hpp>` | 8.2 s | 8.2 s |

The optimization level barely matters, because parsing the dependency headers dominates.
Two things follow. Include the provider header you use rather than `<cail/cail.hpp>`,
which saves about 1.5 s per translation unit. If several translation units call CAIL, give
the project a precompiled header or a unity build, which is where the rest of the time
comes back.

## Examples

Build the example binaries from a checkout:

```sh
cmake -S . -B build
cmake --build build
```

Set the provider's API key in your environment, then run the binaries from `./build`:

| Provider | Key | Examples |
| --- | --- | --- |
| OpenAI | `OPENAI_API_KEY` | `cail_openai_prompt`, `cail_openai_stream`, `cail_openai_generate_object`, `cail_openai_tool_call`, `cail_openai_agent`, `cail_openai_embed`, `cail_openai_create`, `cail_openai_multimodal_memory` |
| OpenRouter | `OPENROUTER_API_KEY` | `cail_openrouter_prompt`, `cail_openrouter_stream`, `cail_openrouter_object` |
| Anthropic | `ANTHROPIC_API_KEY` | `cail_anthropic_prompt`, `cail_anthropic_stream`, `cail_anthropic_object`, `cail_anthropic_tool_call` |
| Gemini | `GEMINI_API_KEY` | `cail_gemini_prompt`, `cail_gemini_stream`, `cail_gemini_object`, `cail_gemini_tool_call`, `cail_gemini_embed` |
| Mistral | `MISTRAL_API_KEY` | `cail_mistral_prompt`, `cail_mistral_embed` |
| Charm Hyper | `HYPER_API_KEY` | `cail_hyper_prompt` |
| Azure Foundry | `AZURE_FOUNDRY_API_KEY` | `cail_foundry_prompt` |
| OpenCode | `OPENCODE_API_KEY` | `cail_opencode_prompt` |
| Ollama Cloud | `OLLAMA_API_KEY` | `cail_ollama_cloud_prompt` |
| Local model | none | `cail_local_prompt`, `cail_local_object`, `cail_local_tool_call` |

For example:

```sh
OPENAI_API_KEY=... ./build/cail_openai_prompt
```

`cail_typed_schema` round-trips an `Analysis` value and prints its generated JSON Schema,
so it needs no key.

## Next steps

- [Quickstart](https://martineastwood.github.io/cail/guides/quickstart/) for a guided first request
- [Providers](https://martineastwood.github.io/cail/guides/providers/) for provider setup and capabilities
- [Agent](https://martineastwood.github.io/cail/guides/agent/) to bundle a model, instructions, and tools
- [Memory](https://martineastwood.github.io/cail/guides/memory/) to give an agent conversation memory
- [Advanced usage](https://martineastwood.github.io/cail/guides/advanced/) for middleware, options, and images
