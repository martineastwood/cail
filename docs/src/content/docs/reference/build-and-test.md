---
title: Build and test
description: Configure, build, test, format, and analyze CAIL with the ./dev script.
---

The `./dev` script wraps configure, build, test, formatting, and static
analysis, so you run the same commands locally that CI runs.

## Full validation

```sh
./dev check
```

Runs everything: formatting, configure, build, unit tests, clang-tidy, and an
ASan/UBSan build and test. Use it for release readiness or major build and
toolchain changes.

## Focused commands

Configure once per build directory, then build and run what you need:

```sh
./dev configure
./dev build build/dev cail_openai_test
./dev test build/dev cail_openai
```

Other unit-test groups are `cail_core`, `cail_chat_completions`, `cail_foundry`,
and `cail_opencode`. `cail_local_http` starts a local Python server, and
`cail_install_smoke` builds a consumer against the installed package.

Run `./dev test -N` to list the available suites.

## Formatting and analysis

```sh
./dev format        # Check formatting. ./dev format --fix rewrites the files.
./dev format --fix
./dev tidy          # clang-tidy over the test translation units.
./dev tidy --fix
./dev sanitizer     # Build and test with ASan and UBSan.
```

Formatting is pinned to `clang-format` 23.1.1. `./dev format` warns when your
local version differs from the one CI uses.

## Examples

Build example targets from a configured build directory:

```sh
cmake --build build --target cail_openai_prompt cail_openai_stream
./build/cail_openai_prompt
```

Set the provider's API key in your environment before running, for example
`OPENAI_API_KEY` for the OpenAI examples and `OPENROUTER_API_KEY` for the
OpenRouter ones.

## Related pages

- [Install](/guides/install/) for dependencies and package installation
- [Compile time](/reference/compile-time/) for header costs
