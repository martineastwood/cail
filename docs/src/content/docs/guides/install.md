---
title: Install
description: Install CAIL as a CMake package, or build and test it from source.
---

CAIL is a header-only C++23 library. You can install it to a prefix with CMake,
or build and test it directly from a source checkout.

You need CMake 3.31 or newer, a C++23 compiler supported by the pinned Glaze
release, and OpenSSL development files for HTTPS transport.

After installing, continue with the [Quickstart](/guides/quickstart/) to make
your first request.

## Install as a CMake package

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

## Dependencies

CAIL builds against [Glaze](https://github.com/stephenberry/glaze) 8.4.0 or
newer and [magic_enum](https://github.com/Neargye/magic_enum). The first CMake
configure downloads them unless you already have them installed.

To build without network access, install them yourself or point CMake at a
prefix that contains them:

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/deps
```

Installing CAIL also installs Glaze and magic_enum into the same prefix, so a
project that consumes the installed package resolves both through
`CMAKE_PREFIX_PATH` alone.

## Build and test from source

The `./dev` script wraps configure, build, test, formatting, and static
analysis. Run everything the way CI does:

```sh
./dev check
```

While you iterate on an adapter, configure once and then build and run the
affected suite:

```sh
./dev configure
./dev build build/dev cail_openai_test
./dev test build/dev cail_openai
```

Other unit-test groups are `cail_core`, `cail_chat_completions`, `cail_foundry`,
and `cail_opencode`. `cail_local_http` starts a local Python server, and
`cail_install_smoke` builds a consumer against the installed package.

Run the remaining checks before you open a pull request:

```sh
./dev format        # Check formatting. ./dev format --fix rewrites the files.
./dev tidy          # clang-tidy over the test translation units.
./dev sanitizer     # Build and test with ASan and UBSan.
```

Formatting is pinned to `clang-format` 23.1.1. `./dev format` warns when your
local version differs from the one CI uses.

## Next steps

- [Quickstart](/guides/quickstart/) for your first generation request
- [Providers](/guides/providers/) to pick and configure a provider
