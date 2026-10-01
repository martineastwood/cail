Follow YAGNI principles, and one-liner solutions. Do not shim or maintain legacy
behavior: this is greenfield, so breaking changes are fine.

## Local feedback loop

Run these commands from the `cail/` folder. Configure once, then build and test
only the affected targets while editing:

```sh
./dev configure
./dev build build/dev cail_openai_test
./dev test build/dev '^cail_openai$'
```

Build before testing: `./dev test` runs existing binaries. For shared code,
build and run every affected suite. `ctest --test-dir build/dev -N` lists
suites. Target names end in `_test`; suite names omit that suffix.

Examples are off through `./dev` (CMake's own default is ON), compilation uses
four jobs by default, and ccache is used when installed. Keep build directories
and caches for incremental builds. Use `JOBS=2` if memory is tight. Do not run
full checks, sanitizers, or the Linux matrix after every edit. Documentation-only
changes need no C++ builds or tests; the docs site is in `docs/`.

## Checkpoints before pushing

- Run `./dev check` for code or build changes. It checks formatting, configures,
  builds native test targets, and runs tests except packaging.
- Run `./dev tidy --changed` before committing C++ changes. After committing,
  name changed source files explicitly, such as `./dev tidy src/openai.cpp`.
  Header, CMake, and clang-tidy configuration changes trigger full analysis.
- Run `./dev linux-check` for C++ or build changes, or a CI-only compile failure.
  It builds all tests and examples and runs every suite with Ubuntu 24.04 GCC.
  Docker is required. A native AppleClang pass does not cover Linux/GCC issues.
- Run `./dev sanitizer` for memory safety changes and `./dev thread-sanitizer`
  for concurrency changes. Both include packaging tests.
- Run `./dev test --all` for native packaging validation when changing
  installation, dependencies, public API, or build settings. Build first.

Do not repeat passing checks unless subsequent changes or unresolved failures
justify it. CI retains full compiler, sanitizer, analysis, and packaging coverage.
If a required tool is unavailable, report the check as unrun.

## Focused Linux and packaging checks

```sh
CAIL_BUILD_EXAMPLES=OFF ./dev linux-check cail_openai_test '^cail_openai$'
./dev test build/dev '^cail_install_smoke$' --all
```

## Structure and boundaries

`include/cail/*.hpp` is the public API, with `cail.hpp` as the umbrella header.
`include/cail/detail/`, `src/detail/`, and the rest of `src/` are internal. Tests
are built with `src/` on their include path, so they may include internals.

One static library, `cail`, linked as `cail::cail`. `CMakeLists.txt` lists sources
by hand: a new provider needs a header, a `src/` file, and an entry in that list.

`cail_add_test(<name>)` in `CMakeLists.txt` compiles `tests/<name>_test.cpp` into
target `cail_<name>_test` and suite `cail_<name>`, so registering a suite is one
line there. `cail_local_http` needs Python; `cail_install_smoke` is the packaging
test and runs only with `--all`. Conan packaging lives in `conanfile.py`,
`test_package/`, and `ports/`.

## Pinned tools and editor setup

CMake 3.31 or newer; the pinned version in `ci/cmake-version.txt` is shared by CI
and the Linux check, which also relies on the compiler launcher and cache wiring.
The first configure fetches Glaze 8.4.0 and magic_enum 0.9.8 unless CMake finds
installed copies. LLVM 23 is the local baseline: `./dev format` warns when
clang-format is not 23.1.1, and `./dev tidy` prefers Homebrew `llvm@23`.

`.clangd` reads `build/compile_commands.json`, but `./dev configure` writes
`build/dev/`, so configure a directory named `build` to get editor diagnostics.

See [Build and test](docs/src/content/docs/reference/build-and-test.md) for the
complete contributor guide.
