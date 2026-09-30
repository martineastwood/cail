#pragma once

#include <cail/detail/glaze_meta.hpp>
#include <cail/error.hpp>

#include <optional>
#include <string>
#include <type_traits>
#include <vector>

namespace cail::detail {

template <typename T> struct ValueValidator {
  static Result<void> check(const T& value, const std::string& path) {
    Result<void> result;
    if constexpr (glz::reflectable<T>) {
      std::size_t index = 0;
      glz::for_each_field(value, [&](const auto& member) {
        if (result)
          result = ValueValidator<std::remove_cvref_t<decltype(member)>>::check(
              member, path + "." + std::string{glz::reflect<T>::keys[index]});
        ++index;
      });
    }
    return result;
  }
};

template <typename T> struct ValueValidator<Field<T>> {
  static Result<void> check(const Field<T>& field, const std::string& path) {
    if constexpr (std::is_arithmetic_v<T> && !std::is_same_v<T, bool>) {
      if (field.minimum && !(field.value >= *field.minimum))
        return std::unexpected(Error{.code = ErrorCode::schema_validation,
                                     .message = path + ": value does not satisfy minimum."});
      if (field.maximum && !(field.value <= *field.maximum))
        return std::unexpected(Error{.code = ErrorCode::schema_validation,
                                     .message = path + ": value does not satisfy maximum."});
    }
    return ValueValidator<T>::check(field.value, path);
  }
};

template <typename T> struct ValueValidator<std::optional<T>> {
  static Result<void> check(const std::optional<T>& value, const std::string& path) {
    return value ? ValueValidator<T>::check(*value, path) : Result<void>{};
  }
};

template <typename T, typename Allocator> struct ValueValidator<std::vector<T, Allocator>> {
  static Result<void> check(const std::vector<T, Allocator>& values, const std::string& path) {
    for (std::size_t i = 0; i < values.size(); ++i) {
      auto result = ValueValidator<T>::check(values[i], path + "[" + std::to_string(i) + "]");
      if (!result)
        return result;
    }
    return {};
  }
};

} // namespace cail::detail
