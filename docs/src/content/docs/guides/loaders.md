---
title: Load files, images, and PDFs
description: Read local files and attach text, images, and PDFs to model requests.
---

You can load a local document or image and send it with your prompt. Include
`<cail/loaders.hpp>` alongside your provider header, or use `<cail/cail.hpp>`.

## Summarize a PDF

Set `OPENAI_API_KEY` and place `report.pdf` in your application's working directory:

```cpp
#include <cail/generate.hpp>
#include <cail/loaders.hpp>
#include <cail/openai.hpp>

#include <iostream>
#include <utility>

int main() {
    auto pdf = cail::load_pdf("report.pdf");
    if (!pdf) {
        std::cerr << pdf.error().message << '\n';
        return 1;
    }

    auto response = cail::generate_text({
        .model = cail::openai("gpt-6-luna"),
        .messages = {cail::Message{
            .content = {
                cail::TextPart{.text = "Summarize the key findings in this report."},
                std::move(*pdf),
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

`load_pdf` returns a `PdfPart` containing the file's bytes and its filename.
Attach it to a user message alongside your question. The provider reads the PDF,
including page images where supported. Loading a PDF does not extract text locally.

PDF inputs work with OpenAI Responses, Anthropic, and Gemini adapters. Azure Foundry
and OpenCode's Responses, Anthropic Messages, and Gemini API families use these
same formats. The selected model or deployment must also accept PDFs. Check
`model.adapter_capabilities().pdf_input` for adapter support. Chat Completions
adapters reject PDF parts.

See the provider guides for PDF model support and request limits:
[OpenAI](https://developers.openai.com/api/docs/guides/file-inputs),
[Anthropic](https://platform.claude.com/docs/en/build-with-claude/pdf-support), and
[Gemini](https://ai.google.dev/gemini-api/docs/document-processing).

## Load text or an image

Use the same message pattern with `load_text` or `load_image`:

```cpp
auto notes = cail::load_text("notes.md");
auto image = cail::load_image("chart.png");
if (!notes || !image) {
    // Handle the loader error before using its value.
    return 1;
}

cail::Message message{
    .content = {
        cail::TextPart{.text = "Explain this chart using the notes."},
        std::move(*notes),
        std::move(*image),
    },
};
```

`load_text` preserves the file's bytes and line endings. Supply text in the encoding
your provider accepts, normally UTF-8. It does not convert Word documents, HTML,
or other document formats to plain text.

`load_image` detects PNG, JPEG, GIF, and WebP from their file signatures and sets
the MIME type. The filename extension does not affect detection. The loader does
not decode, resize, or verify the entire image. Your model must support the
image format you send.

## Keep attachments in a conversation

You can pass the loaded parts to an agent as a user `Message`. Configure
conversation memory and an id to reuse the attachments in follow-up turns:

```cpp
cail::Agent agent({
    .model = cail::openai("gpt-6-luna"),
    .memory = std::make_shared<cail::InMemoryConversationMemory>(),
    .conversation_id = "chart-review",
});
auto first = agent.generate(std::move(message));
if (!first) {
    std::cerr << first.error().message << '\n';
    return 1;
}
auto follow_up = agent.generate("What is the main trend?");
```

Here `message` is the user message from the text-and-image example above.
Include `<cail/agent.hpp>` and `<memory>`. File memory can also keep attachments
across restarts. See [Memory](/guides/memory/) for PDF conversations, streaming,
async calls, and history limits.

## Read raw bytes

Use `load_file` when you need the contents as a `std::string`, for example when
preparing text for an embedding request:

```cpp
auto bytes = cail::load_file("notes.txt");
if (!bytes) {
    std::cerr << bytes.error().message << '\n';
    return 1;
}
// *bytes contains the original file contents, including any NUL bytes.
```

## Limit file size and handle errors

Every loader takes an optional `max_bytes` argument. The default is 32 MiB:

```cpp
auto pdf = cail::load_pdf("report.pdf", 8 * 1024 * 1024);
```

This limit applies to the source file. Provider encoding increases the request
size, and the provider may impose smaller request limits or page limits. Choose
a size that fits your selected provider.

All loaders return `cail::Result<T>`. Missing files, directories, read failures,
files larger than `max_bytes`, and unrecognized image or PDF signatures return
`ErrorCode::file`. Empty text and raw files can be loaded; empty images and PDFs
are rejected. PDF detection checks for a `%PDF-` header at the start of the file,
so a matching header does not guarantee a valid document.

Paths are local and relative paths use your application's working directory.
Loaders do not fetch URLs. PDF parts are accepted only in user messages, and
need nonempty bytes and a filename when you construct them yourself.

## Next steps

- [Providers](../providers/) to configure a model that accepts your inputs.
- [Structured outputs](../structured-output/) to extract document data into C++ types.
- [Embeddings](../embeddings/) to embed loaded text for search.
