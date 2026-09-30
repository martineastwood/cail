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

Numeric bounds are inclusive. Use them on any arithmetic `Field` type (for example
`double`, `int`, or `std::uint64_t`). CAIL checks them locally when decoding JSON
for `generate_object`, `generate_object_async`, and typed tool arguments. A
confidence of `1.5` returns `ErrorCode::schema_validation` with a message such as
`$.confidence: value does not satisfy maximum.` Nested structs and vector elements
are validated recursively, so errors include the property path and array index,
for example `$.results[1].confidence`. The same checks apply to synchronous,
callback, and coroutine (`co_await generate_object_async`) calls.

C++ enums restrict values to their named choices. An unknown choice returns
`ErrorCode::json_deserialization`.

### Limit string and array lengths

You can require a nonempty title and limit the number of highlights:

```cpp
struct Summary {
    cail::Field<std::string> title{
        .min_length = 1,
        .max_length = 120,
    };
    cail::Field<std::vector<std::string>> highlights{
        .min_items = 1,
        .max_items = 5,
    };
};

auto summary = cail::generate_object<Summary>({
    .model = cail::openai("gpt-6-luna"),
    .prompt = "Summarize this release: faster search, saved filters, and CSV export.",
});
if (!summary) {
    std::cerr << summary.error().message << '\n';
}
```

Include `<string>`, `<vector>`, and `<iostream>` alongside your CAIL headers.
All length bounds are inclusive, and zero is allowed. Omit a bound to leave
that limit unrestricted. Use `min_length` and `max_length` on string fields,
and `min_items` and `max_items` on vector fields.

String length counts Unicode code points, rather than UTF-8 bytes or visible
characters. For example, `é` and `😀` each count as one code point, while an
`e` followed by a combining accent counts as two.

CAIL sends these limits as `minLength`, `maxLength`, `minItems`, and `maxItems`
in JSON Schema and checks them locally when decoding typed values. Invalid
values return `ErrorCode::schema_validation`, for example
`$.highlights: value does not satisfy max_items.` These checks also apply to
typed tool arguments and asynchronous generation.

When you define a schema directly rather than through fields, set
`Schema::min_length` and `Schema::max_length` for strings, or `Schema::min_items`
and `Schema::max_items` for arrays. Use `Schema::items` to describe each array
value. These manually configured limits are sent to the provider; CAIL does not
check them locally during typed decoding.

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

Use the same options with `generate_object_async<T>`. Omit the completion callback
and `co_await` the call inside a `cail::Task`, or pass a callback when you
prefer event-driven code:

```cpp
auto analysis = co_await cail::generate_object_async<Analysis>({
    .model = cail::openai("gpt-6-luna"),
    .prompt = "Classify this review: ...",
});
```

With a completion callback:

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

- [Async and coroutines](/guides/async/) to `co_await` typed generation
- [Tools](/guides/tools/) for typed function tool handlers
- [Providers](/guides/providers/) for model support per provider
