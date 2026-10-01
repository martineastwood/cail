---
title: Build and test
description: Get fast feedback while developing CAIL, then validate across compilers in CI.
---

You can build and test just the behavior you are changing. This page is for
contributors working in the CAIL repository. To use CAIL in your own application,
follow [Install](/guides/install/) instead.

## Start with a focused test

Configure once, then build the affected target and run its suite:

```sh
./dev configure
./dev build build/dev cail_openai_test
./dev test build/dev '^cail_openai$'
```

The local configuration disables examples. Compilation and clang-tidy use four
parallel jobs by default. Set `JOBS=2` if you need to reduce memory use, or raise
it if your machine has room. If `ccache` is installed, new configurations use it
automatically. On macOS, you can install it with `brew install ccache`, then run
`./dev configure` again.

Keep your build directory between edits. Build each affected target when you
change shared code. `./dev test` runs existing binaries, so always build first.
A filter that matches no tests returns an error.

Run `ctest --test-dir build/dev -N` to list suites. Test target names end in
`_test`, such as `cail_async_test`; suite names omit that suffix, such as
`cail_async`. `cail_local_http` uses a local Python server.

## Before pushing

```sh
./dev check
./dev tidy --changed
```

`check` checks formatting, configures, builds all native test targets, and runs
the tests except packaging. It uses your machine's compiler. It does not run
clang-tidy or sanitizers.

`tidy --changed` analyzes staged, unstaged, and untracked C++ source files under
`src` and `tests` relative to `HEAD`. Commit your changes after running it.
Changes to library or test headers, `CMakeLists.txt`, or `.clang-tidy` trigger full
analysis because they can affect multiple source files. To analyze particular
files, including changes you have already committed, name them explicitly:

```sh
./dev tidy src/openai.cpp tests/openai_test.cpp
```

CI runs full clang-tidy, GCC, Clang, AppleClang, ASan/UBSan, ThreadSanitizer, and
packaging checks. Configure, build, and test appear as separate steps. Test
reports and logs are available in each run's artifacts, including failed runs.

## Check Linux GCC

For C++ changes, run the Linux check at a checkpoint before pushing, especially
when your native build uses AppleClang:

```sh
./dev linux-check
```

This requires Docker and uses Ubuntu 24.04, GCC, and the pinned CMake version
used by CI. The full command builds examples and all tests, including packaging.
Docker volumes retain the build directory, compiler cache, and dependency
checkouts between runs.

While investigating one suite, use a focused check:

```sh
CAIL_BUILD_EXAMPLES=OFF ./dev linux-check cail_openai_test '^cail_openai$'
```

A focused pass covers only the selected target and suite. Run the full command
for broad changes or before considering a Linux-only failure resolved.

To reset the Linux build and caches:

```sh
docker volume rm cail-linux-check-build cail-linux-check-ccache cail-linux-check-fetchcontent
```

## Packaging and sanitizers

Packaging installs CAIL and builds a separate consumer. Run it when changing
installation, dependencies, public API, or build settings:

```sh
./dev build
./dev test --all
# Or run packaging alone:
./dev test build/dev '^cail_install_smoke$' --all
```

For memory safety or concurrency changes, run the relevant sanitizer:

```sh
./dev sanitizer
./dev thread-sanitizer
```

These use separate build directories and include packaging tests. To configure
without building, pass `--configure-only`. You can then use `./dev build` and
`./dev test` with `build/asan` or `build/tsan` to focus on a target and suite.

## Formatting and full analysis

```sh
./dev format
./dev format --fix
./dev tidy
./dev tidy --fix src/openai.cpp
```

Formatting is pinned to `clang-format` 23.1.1. The command warns when your local
version differs from CI. Full clang-tidy analyzes library sources and tests;
use it for broad changes or when investigating analysis failures.

## Examples

Enable examples explicitly, then build the ones you need:

```sh
CAIL_BUILD_EXAMPLES=ON ./dev configure
cmake --build build/dev --target cail_openai_prompt cail_openai_stream --parallel 4
./build/dev/cail_openai_prompt
```

Set the provider's API key before running an example, such as `OPENAI_API_KEY`
for OpenAI or `OPENROUTER_API_KEY` for OpenRouter.

## Related pages

- [Install](/guides/install/) for dependencies and package installation
- [Compile time](/reference/compile-time/) for header costs
