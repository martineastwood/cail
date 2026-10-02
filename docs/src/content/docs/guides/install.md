---
title: Install
description: Add CAIL to a C++23 project with CMake and OpenSSL.
---

CAIL is a static C++23 library for calling LLM providers from your application.
You install it once with CMake, link `cail::cail` in your project, and include the
headers you need. Include `<cail/cail.hpp>` for the full API, or narrower headers
such as `<cail/generate.hpp>` and `<cail/openai.hpp>` for a single provider.

> **CAIL is in alpha.** Interfaces can change between releases. Pin the version you
> depend on and skim the docs when you upgrade.

## Prerequisites

- CMake 3.31 or newer
- A C++23 compiler supported by the pinned Glaze release
- OpenSSL development files for HTTPS

## Install as a CMake package

From a CAIL source tree, install the library and its dependencies to a prefix:

```sh
cmake -S . -B build -DCAIL_BUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF
cmake --build build
cmake --install build --prefix /path/to/cail
```

In your application's `CMakeLists.txt`:

```cmake
find_package(cail 0.3 CONFIG REQUIRED)
target_link_libraries(app PRIVATE cail::cail)
```

Configure your app with `-DCMAKE_PREFIX_PATH=/path/to/cail` so CMake finds the
installed package. Build CAIL with the same compiler and compatible build
settings as your application.

The install step also places Glaze and magic_enum in the same prefix, so you do
not need separate dependency installs for a normal consumer build.

## Dependencies at configure time

CAIL uses [Glaze](https://github.com/stephenberry/glaze) 8.4.0 or newer and
[magic_enum](https://github.com/Neargye/magic_enum). The first `cmake` configure
in the CAIL tree downloads them unless they are already on your system.

For offline or locked-down builds, install those libraries yourself and point
CMake at them:

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/deps
```

## HTTPS certificates

The default HTTP transport uses Apple's system root certificates on macOS and
your distribution's CA bundle on Linux. Minimal Linux containers need the
`ca-certificates` package installed, even when you link OpenSSL statically.

For a company proxy or private provider, you can supply a PEM certificate bundle
before starting your application:

```sh
SSL_CERT_FILE=/path/to/company-ca-bundle.pem ./my-app
```

`SSL_CERT_FILE` and `SSL_CERT_DIR` replace the default trust roots. Include all
the CAs your application needs. Certificate and hostname verification remain
enabled. Custom macOS Keychain trust settings are not imported automatically;
use a custom bundle for those certificates.

## Try the examples (optional)

If you cloned the repository and want runnable samples before wiring CAIL into
your own target:

```sh
cmake -S . -B build
cmake --build build --target cail_openai_prompt
export OPENAI_API_KEY="your-key"
./build/cail_openai_prompt
```

Set the API key for whichever provider example you run. The [Providers](/guides/providers/)
guide lists environment variable names.

## Contributing to CAIL

If you are changing the library itself, see [Build and test](/reference/build-and-test/)
for the `./dev` workflow, test suites, and CI checks. [Compile time](/reference/compile-time/)
covers header include costs when you add CAIL to a large codebase.

## Next steps

- [Quickstart](/guides/quickstart/) for your first generation request
- [Providers](/guides/providers/) to pick and configure a provider
