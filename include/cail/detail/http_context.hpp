#pragma once

#include <cail/error.hpp>
#include <cail/http.hpp>

#include <glaze/glaze.hpp>

#include <cctype>
#include <optional>
#include <string>

namespace cail::detail {

inline void attach_http_context(Error& error, const HttpResponse& response) {
  error.http_status = response.status_code;
  for (const auto& header : response.headers) {
    std::string name = header.name;
    for (auto& character : name) {
      character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
    }
    if (name == "x-request-id" || name == "request-id") {
      error.request_id = header.value;
      break;
    }
  }
}

template <typename T>
[[nodiscard]] Result<T> unexpected_with_http_context(Error error, const HttpResponse& response) {
  attach_http_context(error, response);
  return std::unexpected(std::move(error));
}

[[nodiscard]] inline bool is_http_error_status(int status_code) {
  return status_code < 200 || status_code >= 300;
}

// Parses provider JSON error bodies with `message` plus optional `code`/`type`/`status`.
[[nodiscard]] inline Error http_status_error_from_json_body(const HttpResponse& response) {
  struct ProviderError {
    std::string message;
    std::optional<std::string> code;
    std::optional<std::string> type;
    std::optional<std::string> status;
  };
  struct ErrorBody {
    std::optional<ProviderError> error;
  };
  ErrorBody body;
  const auto parsed = glz::read<glz::opts{.error_on_unknown_keys = false}>(body, response.body);
  return Error{
      .code = ErrorCode::http_status,
      .message = !parsed && body.error ? body.error->message : response.body,
      .provider_code = !parsed && body.error ? body.error->code.value_or("") : "",
      .provider_type =
          !parsed && body.error ? body.error->type.value_or(body.error->status.value_or("")) : "",
  };
}

} // namespace cail::detail
