---
title: Compile time
description: How CAIL's header-only design affects your project's build times.
---

CAIL is header-only, so each translation unit that includes it also parses
Glaze, Asio, and OpenSSL. This page shows what that costs and the two changes
that give most of the time back.

Measured on an Apple M1 Pro with AppleClang 17, best of three runs:

| Included header | `-O0` | `-O2` |
| --- | --- | --- |
| `<cail/error.hpp>` | 0.1 s | 0.1 s |
| `<cail/generation.hpp>` | 1.2 s | 1.2 s |
| `<cail/openai.hpp>` | 6.7 s | 6.8 s |
| `<cail/cail.hpp>` | 8.2 s | 8.2 s |

The optimization level barely matters, because parsing the dependency headers
dominates. Two things follow:

- Include the provider header you use rather than `<cail/cail.hpp>`, which
  saves about 1.5 s per translation unit.
- If several translation units call CAIL, give the project a precompiled header
  or a unity build, which is where the rest of the time comes back.

## Related pages

- [Build and test](/reference/build-and-test/) for the `./dev` commands
- [Install](/guides/install/) for dependency setup
