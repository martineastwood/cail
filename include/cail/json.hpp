#pragma once

#include <cail/detail/glaze_meta.hpp>
#include <cail/error.hpp>

#include <string>
#include <string_view>
#include <utility>

namespace cail {

template <typename T>
[[nodiscard]] Result<std::string> to_json(const T& value)
{
    auto result = glz::write_json(value);
    if (!result) {
        return std::unexpected(Error{
            .code = ErrorCode::json_serialization,
            .message = "Glaze could not serialize the value.",
        });
    }

    return std::move(*result);
}

template <typename T>
[[nodiscard]] Result<T> from_json(std::string_view json)
{
    T value{};
    constexpr glz::opts options{
        .null_terminated = false,
        .error_on_missing_keys = true,
    };

    if (const auto error = glz::read<options>(value, json); error) {
        return std::unexpected(Error{
            .code = ErrorCode::json_deserialization,
            .message = glz::format_error(error, json),
            .byte_offset = error.count,
        });
    }

    return value;
}

} // namespace cail
