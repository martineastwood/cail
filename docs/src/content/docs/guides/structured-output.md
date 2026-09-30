---
title: Structured outputs
description: Get typed C++ results from a model with CAIL schemas and JSON Schema generation.
---

You can ask a model for a typed C++ value instead of free-form text. Describe
the result with `cail::Field<T>` members, call `cail::generate_object<T>()`,
and CAIL generates the JSON Schema, requests the structured output, and decodes
the reply into your struct.

Models need JSON Schema support for this. OpenAI, Anthropic, Gemini, OpenRouter
with supporting models, and Chat Completions providers that accept JSON Schema
all work.

## Define a schema

Each `cail::Field<T>` member describes one property:

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

`cail::Field<T>::value` holds the runtime value. Read it directly and assign
through the field: `analysis.confidence.value` reads a value and
`analysis.confidence = 0.8` updates it.

JSON conversion emits only the value, while schema inspection reads the
metadata. To emit the generated JSON Schema yourself, call
`cail::json_schema<Analysis>()`.

When you define a schema directly rather than through fields, set
`Schema::min_items` and `Schema::max_items` to limit array lengths, and
`Schema::items` to describe each array value.

## Request the object

```cpp
auto analysis = cail::generate_object<Analysis>({
    .model = cail::openai("gpt-6-luna"),
    .prompt = "Classify this review: ...",
});
```

The result is a `Result<Analysis>`. Check it before use, as shown in the
[Quickstart](/guides/quickstart/).

## Generate an object without blocking

Use the same options with `generate_object_async<T>` and receive a typed result
in your callback:

```cpp
auto started = cail::generate_object_async<Analysis>(
    {
        .model = cail::openai("gpt-6-luna"),
        .prompt = "Classify this review: The delivery was fast and the product works well.",
    },
    [](cail::Result<Analysis> result) {
        if (!result) std::cerr << result.error().message << '\n';
        // Use the decoded Analysis when result succeeds.
    });
if (!started) {
    std::cerr << started.error().message << '\n';
}
```

Keep your application running until completion. The callback receives the same
refusal, incomplete-response, and decoding errors as `generate_object<T>`.
Set `tool_loop.stop` to cancel. See [Advanced usage](/guides/advanced/) for
callback lifetime and execution details.

## Optional properties

`std::optional<T>` fields are optional in CAIL schemas and deserialize from
either a missing property or `null`:

```cpp
struct Answer {
    std::string language;
    std::optional<std::string> region;
};
```

OpenAI strict output and tool schemas require every property, so CAIL marks
optional properties as required and allows `null` in their JSON Schema type.
The model returns `null` when no value is available, and your field stays
empty.

## Next steps

- [Tools](/guides/tools/) for typed function tool handlers
- [Providers](/guides/providers/) for model support per provider
