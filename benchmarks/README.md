# Measure async generation locally

You can compare blocking and async generation against a local server, without an API key.

```sh
cmake -S . -B build/dev
cmake --build build/dev --target cail_async_load
python3 tests/local_http_server.py build/dev/cail_async_load
```

The benchmark sends four blocking requests, then runs batches of 64 requests
through model generation, text generation, a tool loop, and streaming. Generation
endpoints wait one second before responding. Tool handlers take 20 milliseconds;
streams return immediately.

Each batch prints launch time, total time, peak client thread count, and peak
resident memory sampled throughout the batch, including completion. The sampler
adds one thread. The benchmark also checks whether three sequential HTTP requests
reuse one connection.

Run it on the machine where you expect to use CAIL. The local server and network
stack affect the results, so compare runs on the same machine and build settings.
