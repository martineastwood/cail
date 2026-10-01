#pragma once

#include <cail/http.hpp>

#include <memory>

namespace glz {
struct http_client;
}

namespace cail::detail {
class GlazeHttpTransport final : public HttpTransport {
public:
  GlazeHttpTransport();

  [[nodiscard]] Result<HttpResponse> send(const HttpRequest& request,
                                          std::stop_token stop) override;

  void send_async(HttpRequest request, HttpCompletion complete, std::stop_token stop) override;

  void stream_async(HttpRequest request, HttpDataHandler on_data, HttpCompletion complete,
                    std::stop_token stop) override;

  [[nodiscard]] Result<HttpResponse>
  stream(const HttpRequest& request, const HttpDataHandler& on_data, std::stop_token stop) override;

private:
  struct Operation;

  void start_request(HttpRequest request, HttpDataHandler on_data, HttpCompletion complete,
                     const std::stop_token& stop, bool streaming = false);

  [[nodiscard]] static Result<HttpResponse> blocking_callback_error();

  std::shared_ptr<glz::http_client> client_;
};

} // namespace cail::detail
