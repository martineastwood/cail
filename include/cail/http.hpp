#pragma once

#include <cail/error.hpp>

#include <algorithm>
#include <chrono>
#include <functional>
#include <memory>
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

class HttpTransport {
public:
  virtual ~HttpTransport() = default;

  [[nodiscard]] virtual Result<HttpResponse> send(const HttpRequest& request) = 0;
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

  [[nodiscard]] Result<HttpResponse> send(const HttpRequest& request) override {
    auto configured = request;
    configured.timeout = timeout_;
    return transport_->send(configured);
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

  [[nodiscard]] Result<HttpResponse> send(const HttpRequest& request) override {
    return retry([&] { return transport_->send(request); });
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
        [&] { return !received_data && !stop.stop_requested(); });
  }

private:
  template <typename Send, typename CanRetry = decltype([] { return true; })>
  [[nodiscard]] Result<HttpResponse> retry(Send&& send, CanRetry&& can_retry = {}) const {
    auto delay = policy_.initial_delay;
    for (std::size_t attempt = 0;; ++attempt) {
      auto response = send();
      const bool retriable =
          response && (response->status_code == 429 || response->status_code >= 500);
      if (!retriable || attempt == policy_.max_retries || !can_retry()) {
        return response;
      }
      std::this_thread::sleep_for(delay);
      delay = std::min(delay * 2, policy_.max_delay);
    }
  }

  std::unique_ptr<HttpTransport> transport_;
  RetryPolicy policy_;
};

} // namespace cail
