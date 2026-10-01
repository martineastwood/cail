#pragma once

#include <cstddef>
#include <expected>
#include <memory>
#include <string>

namespace cail {

enum class ErrorCode {
  json_serialization,
  json_deserialization,
  schema_validation,
  invalid_configuration,
  unsupported_schema,
  transport,
  cancelled,
  backpressure,
  http_status,
  provider_response,
  refused,
  incomplete_response,
  invalid_tool_call,
  tool_call_required,
  tool_execution,
  tool_not_found,
  tool_loop_limit,
  memory,
  file,
};

struct GenerationResponse;

struct Error {
  ErrorCode code{};
  std::string message;
  std::size_t byte_offset{};
  int http_status{};
  std::string provider_code;
  std::string provider_type;
  std::string request_id;
  // Completed model steps and tool results from a failed tool loop.
  std::shared_ptr<const GenerationResponse> partial_response;
};

template <typename T> using Result = std::expected<T, Error>;

[[nodiscard]] inline Error generation_cancelled_error() {
  return Error{.code = ErrorCode::cancelled, .message = "Generation was cancelled."};
}

} // namespace cail
