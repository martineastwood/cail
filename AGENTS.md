Follow YAGNI principles, and one-liner solutions

Do not shim or maintain legacy beahvior, this is greenfield so we can make breaking changes.

## CI compiler checks

The native `./dev check` only exercises the compiler selected on the current
machine. When changing C++ code or investigating a CI-only compile failure, run
`./dev linux-check` to build all CAIL test and example targets and run the tests
with Ubuntu 24.04 GCC. The command requires Docker. This catches
Linux/GCC and libstdc++ issues that a successful AppleClang build cannot catch.

The pinned CMake version is in `ci/cmake-version.txt`, read by CI and by the
Docker preflight. CI uses ccache for compiler outputs, so retain the compiler
launcher and cache setup when editing the build workflow.
