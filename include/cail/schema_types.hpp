#pragma once

#include <glaze/json/generic_fwd.hpp>

#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace cail {

enum class SchemaType {
  object,
  array,
  string,
  integer,
  number,
  boolean,
  null,
};

struct Schema {
  std::variant<SchemaType, std::vector<SchemaType>> type{SchemaType::string};
  std::optional<std::string> description;
  std::optional<double> minimum;
  std::optional<double> maximum;
  std::optional<std::size_t> min_length;
  std::optional<std::size_t> max_length;
  std::optional<std::vector<std::string>> enum_values;
  std::optional<std::map<std::string, std::shared_ptr<Schema>>> properties;
  std::optional<std::vector<std::string>> required;
  std::shared_ptr<Schema> items;
  std::optional<glz::generic> additional_properties;
  std::optional<std::size_t> min_items;
  std::optional<std::size_t> max_items;
};

} // namespace cail
