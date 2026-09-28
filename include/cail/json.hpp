#pragma once

#include <cail/detail/glaze_meta.hpp>
#include <cail/error.hpp>

#include <string>
#include <string_view>
#include <utility>

namespace cail {

template <typename T> [[nodiscard]] Result<std::string> to_json(const T& value) {
  struct WriteOptions : glz::opts {
    bool escape_control_characters = true;
  };
  auto result = glz::write<WriteOptions{}>(value);
  if (!result) {
    return std::unexpected(Error{
        .code = ErrorCode::json_serialization,
        .message = "Glaze could not serialize the value.",
    });
  }

  return std::move(*result);
}

template <typename T> [[nodiscard]] Result<T> from_json(std::string_view json) {
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

[[nodiscard]] inline Result<std::string> merge_json_objects(std::string_view target_json,
                                                            const glz::generic::object_t& fields) {
  glz::generic target;
  if (const auto error = glz::read_json(target, target_json); error || !target.is_object()) {
    return std::unexpected(Error{
        .code = ErrorCode::json_deserialization,
        .message = "The target value must be a JSON object.",
    });
  }
  for (const auto& [key, value] : fields) {
    target[key] = value;
  }
  auto encoded = to_json(target);
  if (!encoded) {
    return std::unexpected(Error{
        .code = ErrorCode::json_serialization,
        .message = "Glaze could not serialize the merged provider options.",
    });
  }
  return std::move(*encoded);
}

} // namespace cail
