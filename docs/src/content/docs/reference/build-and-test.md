---
title: Build and test
description: Configure, build, test, format, and analyze CAIL when you work on the library.
---

This page is for **contributors** working in the CAIL repository. If you only
use CAIL from your own application, follow [Install](/guides/install/) instead.

The `./dev` script wraps configure, build, test, formatting, and static
analysis. Use `./dev linux-check` to build and test in Ubuntu with GCC when your
regular build uses a different compiler, such as AppleClang on macOS.

## Full validation

```sh
./dev check
./dev tidy
./dev sanitizer
```

Together these run formatting, configure, build, unit tests, clang-tidy, and an
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
`cail_memory`, `cail_opencode`, and `cail_request_controls`. `cail_local_http` starts a local Python
server, and `cail_install_smoke` builds a consumer against the installed
package.

Run `ctest --test-dir build/dev -N` to list the available suites.

## Check the Linux GCC build

On macOS, `./dev check` uses AppleClang. To catch compiler errors that only
appear with the Linux toolchain, run:

```sh
./dev linux-check
```

This command requires Docker. It configures Ubuntu 24.04 with GCC and the same
CMake 3.31.10 release used by CI, then builds the test and example targets
and runs all tests. Docker volumes keep the compiler and
dependency caches between runs. Remove them with
`docker volume rm cail-linux-check-ccache cail-linux-check-fetchcontent`.

## Formatting and analysis

```sh
./dev format        # Check formatting. ./dev format --fix rewrites the files.
./dev format --fix
./dev tidy          # clang-tidy over library sources and tests.
./dev tidy --fix
./dev sanitizer     # Build and test with ASan and UBSan.
./dev thread-sanitizer # Build and test for data races with ThreadSanitizer.
```

Formatting is pinned to `clang-format` 23.1.1. `./dev format` warns when your
local version differs from the one CI uses.

## Examples

Build example targets from a configured build directory:

```sh
cmake --build build/dev --target cail_openai_prompt cail_openai_stream
./build/dev/cail_openai_prompt
```

Set the provider's API key in your environment before running, for example
`OPENAI_API_KEY` for the OpenAI examples and `OPENROUTER_API_KEY` for the
OpenRouter ones.

## Related pages

- [Install](/guides/install/) for dependencies and package installation
- [Compile time](/reference/compile-time/) for header costs
