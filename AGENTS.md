Follow YAGNI principles, and one-liner solutions

Do not shim or maintain legacy beahvior, this is greenfield so we can make breaking changes.

## Local feedback loop

Run these commands from the `cail/` folder. Configure once, then build and test
only the affected targets while editing:

```sh
./dev configure
./dev build build/dev cail_openai_test
./dev test build/dev '^cail_openai$'
```

Build before testing: `./dev test` runs existing binaries. For shared code,
build and run every affected suite. Use `ctest --test-dir build/dev -N` to list
suites. Target names end in `_test`; suite names omit that suffix.

Examples are off locally, compilation uses four jobs by default, and ccache is
used when installed. Keep build directories and caches for incremental builds.
Use `JOBS=2` if memory is tight. Do not run full checks, sanitizers, or the Linux
matrix after every edit. Documentation-only changes need no C++ builds or tests.

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

Use focused Linux checks while investigating a suite, then the full check at the
checkpoint. Linux builds persist in the `cail-linux-check-build` Docker volume.

The pinned CMake version is in `ci/cmake-version.txt`, shared by CI and Docker.
Retain the compiler launcher and cache setup when editing the build workflow.

See [Build and test](docs/src/content/docs/reference/build-and-test.md) for the
complete contributor guide.
