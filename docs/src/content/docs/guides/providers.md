---
title: Providers
description: Configure CAIL's hosted providers, OpenCode services, and local model servers.
---

You write against one generation API, and the provider is a value you pass in.
Most providers read their API key from a known environment variable, and each
has a `create_*` function for passing a key, base URL, or extra headers
explicitly.

## OpenAI

With `OPENAI_API_KEY` set, choose a model and call it:

```cpp
auto response = cail::generate_text({
    .model = cail::openai("gpt-6-luna"),
    .prompt = "Summarize the main idea of this paragraph.",
});
```

The OpenAI provider accepts user images, function tools, and typed structured
output, and streams text and tool calls. For a custom API key or base URL:

```cpp
auto openai = cail::create_openai({
    .api_key = std::getenv("OPENAI_API_KEY"),
    .base_url = "https://api.openai.com/v1",
});
```

## Anthropic

With `ANTHROPIC_API_KEY` set:

```cpp
auto model = cail::anthropic("claude-haiku-4-5-20251001");
auto response = cail::generate_text({
    .model = model,
    .prompt = "Explain one benefit of native C++ applications in one sentence.",
});
```

Anthropic accepts user images, tools, and typed structured output with models
that support JSON Schema. It reports reasoning and cache usage when available.
This adapter rejects developer-role messages and continuation tokens. Use
`cail::create_anthropic({.api_key = key, .max_tokens = 2048})` to set
credentials or the response limit explicitly; `.base_url` and `.headers` are
also available.

## Gemini

With `GEMINI_API_KEY` set:

```cpp
auto result = cail::generate_text({
    .model = cail::gemini("gemini-3.5-flash-lite"),
    .prompt = "Reply with one word: C++.",
});
```

The Gemini adapter accepts user images and tools, supports text streaming and
typed structured output, and reports token usage when Google provides it. Use
`cail::create_gemini({.api_key = key})` to pass the key explicitly; `.base_url`
and `.headers` are also available.

## OpenRouter

With `OPENROUTER_API_KEY` set, choose a routed model and call it directly:

```cpp
auto response = cail::openrouter("openai/gpt-4o-mini").generate(cail::GenerationRequest{
    .messages = {cail::Message{
        .role = cail::MessageRole::user,
        .content = {cail::TextPart{.text = "Summarize this note."}},
    }},
});
```

OpenRouter accepts text, user images, tool definitions, tool history, and typed
structured output with models that support JSON Schema. It streams text, tool
calls, and usage. It does not accept continuation tokens. Pass an explicit key
or extra headers with `cail::create_openrouter({.api_key = key, .headers = headers})`.

## Mistral and Charm Hyper

Set `MISTRAL_API_KEY` or `HYPER_API_KEY`, then choose a model:

```cpp
auto mistral = cail::mistral("mistral-vibe-cli-with-tools");
auto hyper = cail::hyper("deepseek-v4-flash");
```

Both providers use CAIL's Chat Completions features, including text streaming
and tools. Model support for images and structured output varies. Use
`cail::create_mistral({.api_key = key})` or `cail::create_hyper({.api_key = key})`
to pass a key or extra headers explicitly.

## Azure Foundry

Azure Foundry serves each model through its OpenAI Responses-compatible
endpoint. Set `AZURE_FOUNDRY_API_KEY`, then pass the deployment's Responses
endpoint URL and deployment name together:

```cpp
auto model = cail::create_foundry_model({
    .endpoint = "https://<resource>.cognitiveservices.azure.com/openai/responses"
                 "?api-version=2025-04-01-preview",
    .deployment = "my-deployment",
});
```

Copy the endpoint URL and API version shown for your deployment in the Azure
portal. The provider sends the deployment name as the model and supports
streaming, tools, images, and typed structured output the same way the OpenAI
provider does. Pass `.api_key` explicitly when you need to set the key other
than through the environment.

## Ollama Cloud

With `OLLAMA_API_KEY` set:

```cpp
auto answer = cail::generate_text({
    .model = cail::ollama_cloud("gemma4:31b"),
    .prompt = "Reply with exactly OK.",
});
```

Ollama Cloud uses CAIL's Chat Completions features, including streaming and
tools. Use `cail::create_ollama_cloud({.api_key = key})` to pass a key or extra
headers explicitly.

## OpenCode

OpenCode models use different request formats. Choose the API family when you
select a model, and pass a stable session ID from your application for each
conversation:

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

Set `OPENCODE_API_KEY` before running, or pass `.api_key` to `create_opencode`.
Choose `OpenCodeService::zen` or `OpenCodeService::go` for the service. CAIL
supports the `chat_completions`, `responses`, `anthropic_messages`, and
`gemini` API families. Availability depends on the service and model you
select. OpenCode's SystemOne decision endpoint uses a separate request format
and is not available through this text generation provider.

Every OpenCode request includes the value from `session_id` in
`x-opencode-session`. Reuse the same ID for requests in one conversation, and
choose the ID in your application. Requests with an empty session ID return an
`invalid_configuration` error. `generate_text` and tool follow-up requests
preserve the session ID; you can also set it on a `GenerationRequest` directly.

## Local models

Run a model on your own machine and use it through the same generation and
streaming APIs. `cail::local` points at `http://127.0.0.1:8080/v1/chat/completions`,
the default endpoint of `llama-server`:

```cpp
auto answer = cail::generate_text({
    .model = cail::local("qwen3-1.7b"),
    .prompt = "Reply with exactly OK.",
});
```

Pass the endpoint for your runtime with `cail::create_local`. Ollama serves the
same API on its own port:

```cpp
auto model = cail::create_local({.endpoint = "http://127.0.0.1:11434/v1/chat/completions"})(
    "qwen3-1.7b-local:latest");
```

Use `.endpoint` for any OpenAI-compatible server, such as LM Studio, vLLM, or a
`llama-server` started on another host or port. Set `.api_key` when your server
requires one, and `.headers` for extra request headers. The model string is the
name your server reports: for llama.cpp, the name in its startup output; for
Ollama, the name from `ollama list`.

A local server usually needs no API key, so CAIL sends no `Authorization`
header unless you set one. Local servers vary in what they implement, so check
your runtime for streaming, tools, and JSON Schema support.

## Any OpenAI-compatible endpoint

For another Chat Completions endpoint:

```cpp
auto model = cail::create_chat_completions({.endpoint = url, .api_key = key});
```

Set `.request_session_header` when the endpoint routes requests by a session
ID; CAIL sends the value from `GenerationRequest::session_id` in that header.

## Check adapter capabilities

Use `model.adapter_capabilities()` to check what the CAIL integration for that
API can encode, decode, and stream:

```cpp
if (model.adapter_capabilities().tools) {
    // This adapter can attach tool definitions to the request.
}
```

These flags describe adapter support, not model support. For example,
`mistral.adapter_capabilities().image_input == true` means CAIL can pass image
parts through the Mistral Chat Completions integration. It does not mean every
Mistral model accepts images. If you send a feature the adapter supports but
the selected model does not, the provider can still reject the request. Check
the provider's model documentation for model-specific availability.

## Next steps

- [Advanced usage](/guides/advanced/) for provider-specific options and images
- [Quickstart](/guides/quickstart/) for your first request
