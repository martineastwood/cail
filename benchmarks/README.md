# Measure async generation locally

You can compare blocking and async generation against a local server, without an API key.

```sh
cmake -S . -B build/dev
cmake --build build/dev --target cail_async_load
python3 tests/local_http_server.py build/dev/cail_async_load
```

The benchmark sends four blocking requests and then 64 async requests to an endpoint
that waits one second before responding. It prints the time to start and finish the
async requests, the client process's thread count and resident memory before and
during the async calls, and whether three sequential requests reused one HTTP
connection.

Run it on the machine where you expect to use CAIL. The local server and network
stack affect the results, so compare runs on the same machine and build settings.
