#pragma once

#include <cail/async.hpp>
#include <cail/detail/asio.hpp>

namespace cail {

// Tasks are lazy: awaiting or spawning them starts the operation.
template <typename T = void> using Task = asio::awaitable<T>;

// Run one workflow from a synchronous entry point. Existing Asio applications
// can co_await CAIL operations or co_spawn them on their own executor instead.
template <typename T> T run(Task<T> task) {
  asio::io_context context;
  auto result = asio::co_spawn(context, std::move(task), asio::use_future);
  context.run();
  return result.get();
}

namespace detail {

// The bridge owns the completion handler and keeps its executor alive while
// callback-based work is pending. Always post, including inline completions.
template <typename T, typename Start>
Task<Result<T>> await_result(Start start, std::stop_token stop = {}) {
  return asio::async_initiate<decltype(asio::use_awaitable), void(Result<T>)>(
      [start = std::move(start), stop](auto handler) mutable {
        auto executor = asio::get_associated_executor(handler);
        auto slot = asio::get_associated_cancellation_slot(handler);
        std::stop_source cancelled;
        auto linked = std::make_shared<LinkedStop>(stop, cancelled.get_token());
        if (slot.is_connected())
          slot.assign([cancelled](asio::cancellation_type type) mutable {
            if (type != asio::cancellation_type::none)
              cancelled.request_stop();
          });
        auto owned = std::make_shared<decltype(handler)>(std::move(handler));
        auto complete =
            complete_once<T>([owned, executor, slot, linked,
                              work = asio::make_work_guard(executor)](Result<T> result) mutable {
              asio::post(executor, [owned, slot, linked, result = std::move(result)]() mutable {
                slot.clear();
                (*owned)(std::move(result));
              });
            });
        AsyncOptions callbacks{.schedule = [executor](std::function<void()> task) {
          asio::post(executor, std::move(task));
        }};
        auto started = initiate_async<T>(complete, [&](auto done) {
          return start(std::move(done), linked->source.get_token(), std::move(callbacks));
        });
        if (!started)
          complete(std::unexpected(started.error()));
      },
      asio::use_awaitable);
}

} // namespace detail
} // namespace cail
