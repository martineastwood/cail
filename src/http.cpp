#include "detail/glaze_http_transport.hpp"
#include <atomic>
#include <cail/detail/asio.hpp>
#include <cail/detail/system_ca.hpp>
#include <cail/http.hpp>
#include <exception>
#include <functional>
#include <future>
#include <glaze/net/http_client.hpp>
#include <memory>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string>
#include <system_error>
#include <thread>
#include <utility>

namespace cail::detail {

namespace {

inline thread_local bool on_glaze_io_thread = false;

struct GlazeIoRuntime {
  asio::io_context context;
  asio::executor_work_guard<asio::io_context::executor_type> work{asio::make_work_guard(context)};
  std::stop_source stop;
  std::jthread first{[this] {
    on_glaze_io_thread = true;
    context.run();
  }};
  std::jthread second{[this] {
    on_glaze_io_thread = true;
    context.run();
  }};

  ~GlazeIoRuntime() {
    stop.request_stop();
    async_worker_registry().shutdown();
    work.reset();
    first.join();
    second.join();
  }
};

[[nodiscard]] inline GlazeIoRuntime& glaze_io_runtime() {
  (void)async_worker_registry();
  // Register OpenSSL cleanup before the runtime so its TLS workers exit first.
  if (OPENSSL_init_ssl(0, nullptr) != 1) {
    throw std::runtime_error("Could not initialize OpenSSL");
  }
  static GlazeIoRuntime runtime;
  return runtime;
}

} // namespace

struct GlazeHttpTransport::Operation : std::enable_shared_from_this<Operation> {
  std::shared_ptr<glz::http_client> client;
  HttpResponse response;
  std::optional<std::error_code> error;
  std::exception_ptr callback_error;
  std::shared_ptr<glz::http_stream_connection> connection;
  HttpDataHandler on_data;
  HttpCompletion complete;
  std::stop_token stop;
  std::unique_ptr<LinkedStop> linked_stop;
  std::optional<std::stop_callback<std::function<void()>>> cancel;
  bool streaming{};
  bool finished{};

  // All operation state and socket operations run on the client's strand.
  void finish() {
    if (std::exchange(finished, true)) {
      return;
    }
    Result<HttpResponse> result;
    if (stop.stop_requested()) {
      result = std::unexpected(http_request_cancelled_error());
    } else if (callback_error) {
      result = std::unexpected(Error{.code = ErrorCode::transport,
                                     .message = "The HTTP stream callback threw an exception."});
    } else if (error) {
      result = std::unexpected(Error{.code = ErrorCode::transport, .message = error->message()});
    } else {
      result = std::move(response);
    }
    connection.reset();
    cancel.reset();
    complete(std::move(result));
  }

  void start(HttpRequest request) {
    if (finished) {
      return;
    }
    const auto executor = client->get_executor();
    cancel.emplace(stop, [weak = weak_from_this(), executor] {
      asio::post(executor, [weak] {
        if (auto active = weak.lock(); active && !active->finished) {
          if (active->connection) {
            active->connection->disconnect();
          } else {
            active->finish();
          }
        }
      });
    });
    if (stop.stop_requested()) {
      finish();
      return;
    }
    auto self = shared_from_this();
    glz::http_headers headers;
    for (const auto& header : request.headers) {
      headers.add(header.name, header.value);
    }
    glz::stream_request_params_v2 params{
        .method = std::move(request.method),
        .url = std::move(request.url),
        .timeout = request.timeout,
        .strategy = glz::stream_read_strategy::immediate_delivery,
        .body = std::move(request.body),
        .headers = std::move(headers),
        .on_data =
            [self](std::string_view bytes) {
              if (self->finished) {
                return;
              }
              if (!self->streaming || self->response.status_code < 200 ||
                  self->response.status_code >= 300) {
                self->response.body.append(bytes);
              } else if (!self->stop.stop_requested() && !self->callback_error && self->on_data) {
                try {
                  self->on_data(bytes);
                } catch (...) {
                  self->callback_error = std::current_exception();
                }
              }
            },
        .on_error =
            [self](std::error_code error) {
              if (!self->finished) {
                self->error = error;
              }
            },
        .on_progress =
            [self](std::size_t, std::size_t) {
              return !self->stop.stop_requested() && !self->callback_error;
            },
        .on_connect =
            [self](const glz::response& response) {
              if (self->finished) {
                return;
              }
              self->response.status_code = response.status_code;
              for (const auto& [name, value] : response.response_headers) {
                self->response.headers.push_back({.name = name, .value = value});
              }
            },
        .on_disconnect =
            [self] { asio::post(self->client->get_executor(), [self] { self->finish(); }); },
        .status_is_error = [](int) { return false; },
    };
    try {
      connection = client->stream_request_v2(params);
      if (!connection) {
        error = std::make_error_code(std::errc::invalid_argument);
        finish();
      }
    } catch (...) {
      error = std::make_error_code(std::errc::io_error);
      finish();
    }
  }
};

GlazeHttpTransport::GlazeHttpTransport()
    : client_(std::make_shared<glz::http_client>(asio::make_strand(glaze_io_runtime().context))) {
  client_->set_graceful_ssl_shutdown(false);
  if (const auto trust = configure_system_ca(*client_); !trust) {
    throw std::system_error(trust.error(), "Could not load trusted TLS certificates");
  }
}

Result<HttpResponse> GlazeHttpTransport::send(const HttpRequest& request, std::stop_token stop) {
  if (on_glaze_io_thread) {
    return blocking_callback_error();
  }
  auto done = std::make_shared<std::promise<Result<HttpResponse>>>();
  auto future = done->get_future();
  send_async(
      request, [done](Result<HttpResponse> result) { done->set_value(std::move(result)); }, stop);
  return future.get();
}

void GlazeHttpTransport::send_async(HttpRequest request, HttpCompletion complete,
                                    std::stop_token stop) {
  start_request(std::move(request), {}, std::move(complete), stop);
}

void GlazeHttpTransport::stream_async(HttpRequest request, HttpDataHandler on_data,
                                      HttpCompletion complete, std::stop_token stop) {
  start_request(std::move(request), std::move(on_data), std::move(complete), stop, true);
}

Result<HttpResponse> GlazeHttpTransport::stream(const HttpRequest& request,
                                                const HttpDataHandler& on_data,
                                                std::stop_token stop) {
  if (on_glaze_io_thread) {
    return blocking_callback_error();
  }
  auto done = std::make_shared<std::promise<Result<HttpResponse>>>();
  auto future = done->get_future();
  start_request(
      request, on_data, [done](Result<HttpResponse> result) { done->set_value(std::move(result)); },
      stop, true);
  return future.get();
}

void GlazeHttpTransport::start_request(HttpRequest request, HttpDataHandler on_data,
                                       HttpCompletion complete, const std::stop_token& stop,
                                       bool streaming) {
  auto operation = std::make_shared<Operation>();
  operation->client = client_;
  operation->on_data = std::move(on_data);
  operation->complete = complete_once<HttpResponse>(std::move(complete));
  operation->linked_stop = std::make_unique<LinkedStop>(stop, glaze_io_runtime().stop.get_token());
  operation->stop = operation->linked_stop->source.get_token();
  operation->streaming = streaming;
  const auto executor = client_->get_executor();
  asio::post(executor, [operation, request = std::move(request)]() mutable {
    operation->start(std::move(request));
  });
}

Result<HttpResponse> GlazeHttpTransport::blocking_callback_error() {
  return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                               .message =
                                   "Blocking HTTP requests cannot run inside an HTTP callback. "
                                   "Start an async request instead."});
}

} // namespace cail::detail

namespace cail {

[[nodiscard]] std::unique_ptr<HttpTransport> make_default_http_transport(TransportOptions options) {
  return std::make_unique<RetryingHttpTransport>(
      std::make_unique<TimeoutHttpTransport>(std::make_unique<detail::GlazeHttpTransport>(),
                                             options.timeout),
      options.retry);
}

} // namespace cail
