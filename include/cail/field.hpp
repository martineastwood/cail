#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <utility>

namespace cail {

template <typename T> struct Field {
  T value{};
  std::string description;
  std::optional<T> minimum;
  std::optional<T> maximum;
  std::optional<std::size_t> min_length;
  std::optional<std::size_t> max_length;
  std::optional<std::size_t> min_items;
  std::optional<std::size_t> max_items;

  Field& operator=(T new_value) {
    value = std::move(new_value);
    return *this;
  }
};

} // namespace cail
