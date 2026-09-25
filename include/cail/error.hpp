#pragma once

#include <cstddef>
#include <expected>
#include <string>

namespace cail {

enum class ErrorCode {
    json_serialization,
    json_deserialization,
    invalid_configuration,
    unsupported_schema,
    transport,
    http_status,
    provider_response,
    refused,
    incomplete_response,
    invalid_tool_call,
    tool_call_required,
    tool_execution,
    tool_not_found,
    tool_loop_limit,
};

struct Error {
    ErrorCode code{};
    std::string message;
    std::size_t byte_offset{};
    int http_status{};
};

template <typename T>
using Result = std::expected<T, Error>;

} // namespace cail
