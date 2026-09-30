#pragma once

#include <cail/http.hpp>
#include <cail/language_model.hpp>

#include <utility>

namespace cail::detail {

template <typename Decode>
[[nodiscard]] Result<void> send_generation_async(HttpTransport& transport,
                                                 GenerationRequest request, HttpRequest http,
                                                 LanguageModel::GenerationCompletion complete,
                                                 std::stop_token stop, Decode decode) {
  if (!complete)
    return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                 .message = "Async generation requires a completion handler."});
  if (stop.stop_requested())
    return std::unexpected(generation_cancelled_error());
  transport.send_async(
      std::move(http),
      [request = std::move(request), complete = std::move(complete), stop,
       decode = std::move(decode)](Result<HttpResponse> response) {
        if (stop.stop_requested()) {
          complete(std::unexpected(generation_cancelled_error()));
        } else if (!response) {
          complete(std::unexpected(response.error()));
        } else if (auto middleware = run_after_response(request, *response); !middleware) {
          complete(std::unexpected(middleware.error()));
        } else if (stop.stop_requested()) {
          complete(std::unexpected(generation_cancelled_error()));
        } else {
          complete(decode(*response));
        }
      },
      stop);
  return {};
}

} // namespace cail::detail
