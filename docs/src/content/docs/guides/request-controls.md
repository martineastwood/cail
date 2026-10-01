---
title: Request controls and results
description: Tune generation, choose tools, and track finish reasons and usage across model steps.
---

You can tune generation with the same options across providers, then inspect why
it stopped and how many tokens each model step used.

## Quickstart

[Install CAIL](/guides/install/) and set `GEMINI_API_KEY`. This example generates
three tips with a token limit, a temperature, and a stop sequence:

```cpp
#include <cail/cail.hpp>

#include <iostream>

int main() {
  const auto response = cail::generate_text({
      .model = cail::gemini("gemini-2.5-flash"),
      .prompt = "Give me three concise tips for learning C++.",
      .max_output_tokens = 512,
      .temperature = 0.4,
      .stop_sequences = {"END"},
      .tool_choice = cail::ToolChoice{.mode = cail::ToolChoiceMode::none},
  });
  if (!response) {
    std::cerr << response.error().message << '\n';
    return 1;
  }
  std::cout << response->text << '\n';
  if (response->finish_reason == cail::FinishReason::length) {
    std::cerr << "The model reached its output limit.\n";
  }
  for (const auto& step : response->steps) {
    if (step.usage) {
      std::cout << "Step " << step.step << ": " << step.usage->input_tokens << " input, "
                << step.usage->output_tokens << " output tokens\n";
    }
  }
  if (response->total_usage) {
    std::cout << "Reported total: " << response->total_usage->input_tokens << " input, "
              << response->total_usage->output_tokens << " output tokens\n";
  }
}
```

The example is available as `examples/request_controls.cpp`. From the repository
root, you can build and run it:

```sh
cmake -S . -B build
cmake --build build --target cail_request_controls
./build/cail_request_controls
```

`generate_text`, `stream_text`, and `generate_text_async` accept these controls.
You can also set them on `GenerationRequest` for direct model calls or an agent.

## Tune generation

Leave an option unset to use the provider's default.

| Option | What you can control |
| --- | --- |
| `max_output_tokens` | Maximum output tokens per model step, greater than zero. |
| `temperature` | Sampling temperature, from 0 to 2, or 0 to 1 for Anthropic. |
| `top_p` | Sampling probability threshold, from 0 to 1. |
| `stop_sequences` | Nonempty strings that stop generation when encountered. |
| `tool_choice` | Whether the model may call tools, must call one, or must call a named tool. |

Many models work best with either `temperature` or `top_p`, not both tuned at
once. CAIL lets you set both when you need to; support varies by model and
provider. An endpoint that rejects an option returns a provider error; CAIL does
not retry by dropping the option.

OpenAI Responses, including Azure Foundry and OpenCode's Responses adapter, does
not support `stop_sequences`. CAIL returns `invalid_configuration` before
sending the request. Chat Completions accepts up to four sequences, Gemini
accepts up to five, and Anthropic accepts a list without a CAIL count limit.

CAIL rejects invalid numeric ranges, empty stop strings, and conflicting common
fields in `provider_options`. Set each control in one place.

## Choose tools

Use `ToolChoiceMode::auto_` to let the model choose, or `none` to disable tool
calls. With [registered tools](/guides/tools/), `required` forces a tool call and
`named` selects a specific tool:

```cpp
.tool_choice = cail::ToolChoice{
    .mode = cail::ToolChoiceMode::named,
    .name = "get_weather",
},
```

The name must match a registered tool. `required` and `named` need at least one
tool. In the automatic tool loop, either choice applies to the first model step;
subsequent steps use `auto_` so the model can answer after receiving tool results.
`auto_`, `none`, sampling controls, stop sequences, and the output limit persist
across follow-up steps. Direct model calls send exactly the choice you provide.

## Read finish reasons

`finish_reason` describes the final model step. `raw_finish_reason` preserves
the provider's original reason, when reported.

| `FinishReason` | Meaning |
| --- | --- |
| `stop` | The model finished normally or reached a stop sequence. |
| `length` | The model reached its output token limit. |
| `tool_calls` | The model requested tools. |
| `content_filter` | The provider refused or filtered the response. |
| `other` | The provider reported another reason, retained in `raw_finish_reason`. |
| `unknown` | The provider did not report a reason. |

OpenAI Responses reports a response status rather than a finish reason. CAIL
uses its incomplete reason when available, otherwise its status, and identifies
tool calls or refusals from the response.

`GenerationStatus` remains useful for checking whether text is complete:
`completed`, `incomplete`, or `refused`. An unfamiliar finish reason maps to
`other` and `incomplete`. The automatic tool loop stops on incomplete or refused
responses and leaves any tool calls unexecuted.

## Track token usage

For `generate_text`, `stream_text`, agent calls, and automatic tool loops:

- `usage` contains usage for the final model call.
- `steps` preserves each model call's text, reasoning, tool calls, finish reason,
  original reason, and optional usage. Step numbers start at zero by default.
- `total_usage` sums the usage reported by all steps. It is absent when no step
  reports usage. If some steps omit usage, the total covers only reported steps.

Direct model calls expose their own `usage` and finish reason. Use the high-level
generation functions when you want `steps` and `total_usage`.

Input counts include reported cache reads and writes. Output counts include
reported reasoning tokens. `cache_read_tokens`, `cache_write_tokens`, and
`reasoning_tokens` are optional breakdowns, so do not add them to the totals
again. An absent breakdown means it was not reported, while zero is a reported
count. Totals count repeated input across tool rounds; they are not counts of
unique tokens in the conversation.

Streaming `UsageUpdate` events describe the current model call. Use the returned
response's `total_usage` for the whole operation. For Chat Completions,
`stream_usage = false` disables requested usage reporting in the stream.

## Next steps

- [Tools](/guides/tools/) for automatic tool execution and stop conditions.
- [Streaming](/guides/streaming/) for deltas and cancellation.
- [Advanced usage](/guides/advanced/) for middleware and provider-specific options.
