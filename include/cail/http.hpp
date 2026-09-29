#pragma once

#include <cail/error.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace cail {

struct HttpHeader {
  std::string name;
  std::string value;
};

struct HttpRequest {
  std::string method{"POST"};
  std::string url;
  std::vector<HttpHeader> headers;
  std::string body;
  std::chrono::seconds timeout{30};
};

struct HttpResponse {
  int status_code{};
  std::vector<HttpHeader> headers;
  std::string body;
};

using HttpDataHandler = std::function<void(std::string_view)>;
using HttpCompletion = std::function<void(Result<HttpResponse>)>;

namespace detail {

[[nodiscard]] inline Error http_request_cancelled_error() {
  return Error{.code = ErrorCode::cancelled, .message = "The HTTP request was cancelled."};
}

class AsyncRetryTimer {
public:
  AsyncRetryTimer() : worker_([this](std::stop_token stop) { run(stop); }) {}
  ~AsyncRetryTimer() {
    worker_.request_stop();
    ready_.notify_all();
  }

  [[nodiscard]] std::size_t schedule(std::chrono::milliseconds delay, std::function<void()> task) {
    std::size_t id;
    {
      std::lock_guard lock(mutex_);
      id = ++next_id_;
      tasks_.push_back({id, std::chrono::steady_clock::now() + delay, std::move(task)});
    }
    ready_.notify_one();
    return id;
  }

  void cancel(std::size_t id) {
    std::lock_guard lock(mutex_);
    std::erase_if(tasks_, [id](const Task& task) { return task.id == id; });
  }

private:
  struct Task {
    std::size_t id;
    std::chrono::steady_clock::time_point due;
    std::function<void()> run;
  };

  void run(std::stop_token stop) {
    std::unique_lock lock(mutex_);
    while (!stop.stop_requested()) {
      if (tasks_.empty()) {
        ready_.wait(lock);
        continue;
      }
      auto next = std::ranges::min_element(tasks_, {}, &Task::due);
      if (ready_.wait_until(lock, next->due) != std::cv_status::timeout || tasks_.empty()) {
        continue;
      }
      next = std::ranges::min_element(tasks_, {}, &Task::due);
      if (next->due > std::chrono::steady_clock::now()) {
        continue;
      }
      auto task = std::move(next->run);
      tasks_.erase(next);
      lock.unlock();
      task();
      lock.lock();
    }
  }

  std::mutex mutex_;
  std::condition_variable ready_;
  std::vector<Task> tasks_;
  std::size_t next_id_{};
  std::jthread worker_;
};

[[nodiscard]] inline AsyncRetryTimer& async_retry_timer() {
  static AsyncRetryTimer timer;
  return timer;
}

} // namespace detail

// Implementations must permit concurrent send and stream calls on the same instance.
class HttpTransport {
public:
  virtual ~HttpTransport() = default;

  [[nodiscard]] virtual Result<HttpResponse> send(const HttpRequest& request,
                                                  std::stop_token stop) = 0;
  virtual void send_async(HttpRequest, HttpCompletion complete, std::stop_token) {
    complete(
        std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                              .message = "The HTTP transport does not support async requests."}));
  }
  // Delivers response body bytes incrementally; the returned body is not buffered.
  [[nodiscard]] virtual Result<HttpResponse>
  stream(const HttpRequest& request, const HttpDataHandler& on_data, std::stop_token stop) = 0;
};

struct RetryPolicy {
  std::size_t max_retries{2};
  std::chrono::milliseconds initial_delay{500};
  std::chrono::milliseconds max_delay{8'000};
};

struct TransportOptions {
  std::chrono::seconds timeout{30};
  RetryPolicy retry;
};

class TimeoutHttpTransport final : public HttpTransport {
public:
  TimeoutHttpTransport(std::unique_ptr<HttpTransport> transport, std::chrono::seconds timeout)
      : transport_(std::move(transport)), timeout_(timeout) {}

  [[nodiscard]] Result<HttpResponse> send(const HttpRequest& request,
                                          std::stop_token stop) override {
    auto configured = request;
    configured.timeout = timeout_;
    return transport_->send(configured, stop);
  }

  void send_async(HttpRequest request, HttpCompletion complete, std::stop_token stop) override {
    request.timeout = timeout_;
    transport_->send_async(std::move(request), std::move(complete), stop);
  }

  [[nodiscard]] Result<HttpResponse> stream(const HttpRequest& request,
                                            const HttpDataHandler& on_data,
                                            std::stop_token stop) override {
    auto configured = request;
    configured.timeout = timeout_;
    return transport_->stream(configured, on_data, stop);
  }

private:
  std::unique_ptr<HttpTransport> transport_;
  std::chrono::seconds timeout_;
};

class RetryingHttpTransport final : public HttpTransport {
public:
  explicit RetryingHttpTransport(std::unique_ptr<HttpTransport> transport, RetryPolicy policy = {})
      : transport_(std::move(transport)), policy_(policy) {}

  [[nodiscard]] Result<HttpResponse> send(const HttpRequest& request,
                                          std::stop_token stop) override {
    return retry([&] { return transport_->send(request, stop); }, stop);
  }

  void send_async(HttpRequest request, HttpCompletion complete, std::stop_token stop) override {
    struct Operation : std::enable_shared_from_this<Operation> {
      Operation(HttpTransport* transport, HttpRequest request, HttpCompletion complete,
                std::stop_token stop, RetryPolicy policy)
          : transport(transport), request(std::move(request)), complete(std::move(complete)),
            stop(stop), policy(policy), delay(policy.initial_delay) {}

      HttpTransport* transport;
      HttpRequest request;
      HttpCompletion complete;
      std::stop_token stop;
      RetryPolicy policy;
      std::size_t attempt{};
      std::chrono::milliseconds delay;
      std::atomic<bool> finished{false};
      std::mutex timer_mutex;
      std::optional<std::size_t> timer_id;
      std::optional<std::stop_callback<std::function<void()>>> cancel;

      void finish(Result<HttpResponse> result) {
        if (finished.exchange(true)) {
          return;
        }
        {
          std::lock_guard lock(timer_mutex);
          if (timer_id) {
            detail::async_retry_timer().cancel(*timer_id);
          }
        }
        complete(std::move(result));
      }

      void start() {
        if (finished.load()) {
          return;
        }
        transport->send_async(
            request,
            [self = shared_from_this()](Result<HttpResponse> result) {
              if (self->finished.load()) {
                return;
              }
              const bool retriable =
                  result && (result->status_code == 429 || result->status_code >= 500);
              if (!retriable || self->attempt == self->policy.max_retries) {
                self->finish(std::move(result));
                return;
              }
              ++self->attempt;
              const auto delay = self->delay;
              self->delay = std::min(self->delay * 2, self->policy.max_delay);
              const auto id =
                  detail::async_retry_timer().schedule(delay, [self] { self->start(); });
              std::lock_guard lock(self->timer_mutex);
              if (self->finished.load()) {
                detail::async_retry_timer().cancel(id);
              } else {
                self->timer_id = id;
              }
            },
            stop);
      }
    };

    auto operation = std::make_shared<Operation>(transport_.get(), std::move(request),
                                                 std::move(complete), stop, policy_);
    operation->cancel.emplace(stop, [weak = std::weak_ptr(operation)] {
      if (auto active = weak.lock()) {
        active->finish(std::unexpected(detail::http_request_cancelled_error()));
      }
    });
    operation->start();
  }

  [[nodiscard]] Result<HttpResponse> stream(const HttpRequest& request,
                                            const HttpDataHandler& on_data,
                                            std::stop_token stop) override {
    bool received_data = false;
    return retry(
        [&] {
          return transport_->stream(
              request,
              [&](std::string_view data) {
                received_data = true;
                if (on_data) {
                  on_data(data);
                }
              },
              stop);
        },
        stop, [&] { return !received_data; });
  }

private:
  template <typename Send, typename CanRetry = decltype([] { return true; })>
  [[nodiscard]] Result<HttpResponse> retry(Send&& send, std::stop_token stop,
                                           CanRetry&& can_retry = {}) const {
    auto delay = policy_.initial_delay;
    for (std::size_t attempt = 0;; ++attempt) {
      if (stop.stop_requested()) {
        return std::unexpected(detail::http_request_cancelled_error());
      }
      auto response = send();
      if (stop.stop_requested()) {
        return std::unexpected(detail::http_request_cancelled_error());
      }
      const bool retriable =
          response && (response->status_code == 429 || response->status_code >= 500);
      if (!retriable || attempt == policy_.max_retries || !can_retry()) {
        return response;
      }
      std::mutex mutex;
      std::unique_lock lock(mutex);
      std::condition_variable_any condition;
      condition.wait_for(lock, stop, delay, [] { return false; });
      delay = std::min(delay * 2, policy_.max_delay);
    }
  }

  std::unique_ptr<HttpTransport> transport_;
  RetryPolicy policy_;
};

} // namespace cail
