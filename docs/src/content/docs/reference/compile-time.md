---
title: Compile time
description: Keep CAIL build times manageable in your C++ project.
---

You can compile CAIL once and reuse it across your application's source files.
Link the static library and include the provider header you use:

```cmake
find_package(cail 0.3 CONFIG REQUIRED)
target_link_libraries(app PRIVATE cail::cail)
```

```cpp
#include <cail/openai.hpp>

auto model = cail::openai("gpt-6-luna");
```

Build CAIL with the same compiler and compatible build settings as your
application. A clean build compiles the library first; subsequent application
builds reuse it until its sources or build configuration change.

Typed JSON, schema generation, and typed tools still compile for the C++ types
you use. The coroutine API exposes Asio, so model headers still include it.

## Choose your headers

- Include the provider header you need, such as `<cail/openai.hpp>`.
- Include `<cail/schema.hpp>` when generating a schema from a C++ type.
- Include `<cail/tool.hpp>` when using `make_tool<T>()` or typed executable tools.
- Include `<cail/cail.hpp>` for the complete API.

For large projects, precompiled headers can reduce the remaining header costs.
Measure a representative clean build and an incremental edit before choosing
additional build optimizations.

## Related pages

- [Build and test](/reference/build-and-test/) for the `./dev` commands
- [Install](/guides/install/) for dependency setup
