#pragma once

#include <cail/http.hpp>

#include <glaze/net/http_client.hpp>

#include <atomic>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string>
#include <system_error>
#include <thread>
#include <utility>

namespace cail::detail {

struct GlazeIoRuntime {
  glz::asio::io_context context;
  glz::asio::executor_work_guard<glz::asio::io_context::executor_type> work{
      glz::asio::make_work_guard(context)};
  std::jthread first{[this] { context.run(); }};
  std::jthread second{[this] { context.run(); }};

  ~GlazeIoRuntime() {
    work.reset();
    context.stop();
  }
};

[[nodiscard]] inline GlazeIoRuntime& glaze_io_runtime() {
  static GlazeIoRuntime runtime;
  return runtime;
}

class GlazeHttpTransport final : public HttpTransport {
public:
  GlazeHttpTransport()
      : client_(std::make_shared<glz::http_client>(glaze_io_runtime().context.get_executor())) {
    client_->set_graceful_ssl_shutdown(false);
  }

  [[nodiscard]] Result<HttpResponse> send(const HttpRequest& request,
                                          std::stop_token stop) override {
    auto done = std::make_shared<std::promise<Result<HttpResponse>>>();
    auto future = done->get_future();
    send_async(
        request,
        [done](Result<HttpResponse> result) { done->set_value(std::move(result)); }, stop);
    return future.get();
  }

  void send_async(HttpRequest request, HttpCompletion complete, std::stop_token stop) override {
    if (stop.stop_requested()) {
      complete(std::unexpected(http_request_cancelled_error()));
      return;
    }

    struct State {
      std::mutex mutex;
      HttpResponse response;
      std::optional<std::error_code> error;
      std::shared_ptr<glz::http_stream_connection> connection;
      HttpCompletion complete;
      std::stop_token stop;
      std::optional<std::stop_callback<std::function<void()>>> cancel;
      std::atomic<bool> finished{false};

      void finish() {
        if (finished.exchange(true)) {
          return;
        }
        Result<HttpResponse> result;
        {
          std::lock_guard lock(mutex);
          if (stop.stop_requested()) {
            result = std::unexpected(http_request_cancelled_error());
          } else if (error) {
            result =
                std::unexpected(Error{.code = ErrorCode::transport, .message = error->message()});
          } else {
            result = std::move(response);
          }
          connection.reset();
        }
        complete(std::move(result));
      }
    };

    auto state = std::make_shared<State>();
    state->complete = std::move(complete);
    state->stop = stop;
    glz::http_headers headers;
    for (const auto& header : request.headers) {
      headers.add(header.name, header.value);
    }
    const auto executor = client_->get_executor();
    glz::stream_request_params_v2 params{
        .method = std::move(request.method),
        .url = std::move(request.url),
        .timeout = request.timeout,
        .strategy = glz::stream_read_strategy::immediate_delivery,
        .body = std::move(request.body),
        .headers = std::move(headers),
        .on_data =
            [state](std::string_view bytes) {
              std::lock_guard lock(state->mutex);
              state->response.body.append(bytes);
            },
        .on_error =
            [state](std::error_code error) {
              std::lock_guard lock(state->mutex);
              state->error = error;
            },
        .on_progress = [stop](std::size_t, std::size_t) { return !stop.stop_requested(); },
        .on_connect =
            [state](const glz::response& response) {
              std::lock_guard lock(state->mutex);
              state->response.status_code = response.status_code;
              for (const auto& [name, value] : response.response_headers) {
                state->response.headers.push_back({.name = name, .value = value});
              }
            },
        .on_disconnect =
            [state, executor] { glz::asio::post(executor, [state] { state->finish(); }); },
        .status_is_error = [](int) { return false; },
    };

    auto connection = client_->stream_request_v2(params);
    if (!connection) {
      {
        std::lock_guard lock(state->mutex);
        state->error = std::make_error_code(std::errc::invalid_argument);
      }
      state->finish();
      return;
    }
    {
      std::lock_guard lock(state->mutex);
      if (!state->finished.load()) {
        state->connection = connection;
      }
    }
    state->cancel.emplace(stop, [connection = std::weak_ptr(connection)] {
      if (auto active = connection.lock()) {
        active->disconnect();
      }
    });
  }

  [[nodiscard]] Result<HttpResponse> stream(const HttpRequest& request,
                                            const HttpDataHandler& on_data,
                                            std::stop_token stop) override {
    if (stop.stop_requested()) {
      return std::unexpected(
          Error{.code = ErrorCode::cancelled, .message = "The HTTP stream was cancelled."});
    }
    struct State {
      HttpResponse response;
      std::optional<std::error_code> error;
      std::exception_ptr callback_error;
      std::promise<void> disconnected;
      std::once_flag finish;
    };

    auto state = std::make_shared<State>();
    auto done = state->disconnected.get_future();
    glz::http_headers headers;
    for (const auto& header : request.headers) {
      headers.add(header.name, header.value);
    }

    glz::stream_request_params_v2 params{
        .method = request.method,
        .url = request.url,
        .timeout = request.timeout,
        .strategy = glz::stream_read_strategy::immediate_delivery,
        .body = request.body,
        .headers = std::move(headers),
        .on_data =
            [state, on_data, stop](std::string_view bytes) {
              if (state->response.status_code >= 400) {
                state->response.body.append(bytes);
                return;
              }
              if (!stop.stop_requested() && !state->callback_error && on_data) {
                try {
                  on_data(bytes);
                } catch (...) {
                  state->callback_error = std::current_exception();
                }
              }
            },
        .on_error =
            [state](std::error_code error) {
              if (state->response.status_code < 400) {
                state->error = error;
              }
            },
        .on_progress =
            [state, stop](std::size_t, std::size_t) {
              return !stop.stop_requested() && !state->callback_error;
            },
        .on_connect =
            [state](const glz::response& response) {
              state->response.status_code = response.status_code;
              state->response.headers.reserve(response.response_headers.size());
              for (const auto& [name, value] : response.response_headers) {
                state->response.headers.push_back(HttpHeader{.name = name, .value = value});
              }
            },
        .on_disconnect =
            [state] {
              std::call_once(state->finish, [state] { state->disconnected.set_value(); });
            },
        .status_is_error = [](int) { return false; },
    };

    auto connection = client_->stream_request_v2(params);
    if (!connection) {
      return std::unexpected(Error{
          .code = ErrorCode::transport,
          .message = "Glaze could not start the HTTP stream.",
      });
    }
    std::stop_callback cancel_on_stop(stop, [connection] { connection->disconnect(); });
    done.wait();
    if (stop.stop_requested()) {
      return std::unexpected(
          Error{.code = ErrorCode::cancelled, .message = "The HTTP stream was cancelled."});
    }
    if (state->callback_error) {
      return std::unexpected(Error{
          .code = ErrorCode::transport,
          .message = "The HTTP stream callback threw an exception.",
      });
    }
    if (state->error) {
      return std::unexpected(Error{
          .code = ErrorCode::transport,
          .message = state->error->message(),
      });
    }
    return std::move(state->response);
  }

private:
  std::shared_ptr<glz::http_client> client_;
};

} // namespace cail::detail

namespace cail {

[[nodiscard]] inline std::unique_ptr<HttpTransport>
make_default_http_transport(TransportOptions options = {}) {
  return std::make_unique<RetryingHttpTransport>(
      std::make_unique<TimeoutHttpTransport>(std::make_unique<detail::GlazeHttpTransport>(),
                                             options.timeout),
      options.retry);
}

} // namespace cail
