#pragma once

#include <cail/detail/glaze_meta.hpp>
#include <cail/field.hpp>
#include <cail/json.hpp>

#include <magic_enum/magic_enum.hpp>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
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
    std::optional<std::vector<std::string>> enum_values;
    std::optional<std::map<std::string, std::shared_ptr<Schema>>> properties;
    std::optional<std::vector<std::string>> required;
    std::shared_ptr<Schema> items;
    std::optional<bool> additional_properties;
};

namespace detail {

template <typename T>
struct field_traits;

template <typename T>
struct field_traits<Field<T>> {
    using value_type = T;
};

template <typename T>
inline constexpr bool is_field_v = false;

template <typename T>
inline constexpr bool is_field_v<Field<T>> = true;

template <typename T>
struct optional_traits;

template <typename T>
struct optional_traits<std::optional<T>> {
    using value_type = T;
};

template <typename T>
inline constexpr bool is_optional_v = false;

template <typename T>
inline constexpr bool is_optional_v<std::optional<T>> = true;

template <typename T>
struct vector_traits;

template <typename T, typename Allocator>
struct vector_traits<std::vector<T, Allocator>> {
    using value_type = T;
};

template <typename T>
inline constexpr bool is_vector_v = false;

template <typename T, typename Allocator>
inline constexpr bool is_vector_v<std::vector<T, Allocator>> = true;

template <typename T>
inline constexpr bool dependent_false_v = false;

template <typename T>
[[nodiscard]] constexpr bool member_is_required()
{
    return !is_optional_v<T>;
}

template <typename T>
[[nodiscard]] Schema make_schema(T& value)
{
    using Value = std::remove_cvref_t<T>;

    if constexpr (is_field_v<Value>) {
        using Underlying = typename field_traits<Value>::value_type;
        auto result = make_schema(value.value);
        if (!value.description.empty()) {
            result.description = value.description;
        }

        if constexpr (std::is_arithmetic_v<Underlying> &&
                      !std::is_same_v<Underlying, bool>) {
            if (value.minimum) {
                result.minimum = static_cast<double>(*value.minimum);
            }
            if (value.maximum) {
                result.maximum = static_cast<double>(*value.maximum);
            }
        }

        return result;
    }
    else if constexpr (is_optional_v<Value>) {
        using Underlying = typename optional_traits<Value>::value_type;
        Schema result;
        if (value) {
            result = make_schema(*value);
        } else {
            Underlying default_value{};
            result = make_schema(default_value);
        }

        if (auto* type = std::get_if<SchemaType>(&result.type)) {
            result.type = std::vector<SchemaType>{*type, SchemaType::null};
        } else {
            auto& types = std::get<std::vector<SchemaType>>(result.type);
            if (std::find(types.begin(), types.end(), SchemaType::null) == types.end()) {
                types.emplace_back(SchemaType::null);
            }
        }
        return result;
    }
    else if constexpr (std::is_same_v<Value, std::string> ||
                       std::is_same_v<Value, std::string_view>) {
        return Schema{.type = SchemaType::string};
    }
    else if constexpr (std::is_same_v<Value, bool>) {
        return Schema{.type = SchemaType::boolean};
    }
    else if constexpr (std::is_integral_v<Value>) {
        return Schema{.type = SchemaType::integer};
    }
    else if constexpr (std::is_floating_point_v<Value>) {
        return Schema{.type = SchemaType::number};
    }
    else if constexpr (std::is_enum_v<Value>) {
        Schema result{.type = SchemaType::string};
        result.enum_values.emplace();
        for (const auto name : magic_enum::enum_names<Value>()) {
            result.enum_values->emplace_back(name);
        }
        return result;
    }
    else if constexpr (is_vector_v<Value>) {
        using Element = typename vector_traits<Value>::value_type;
        Schema result{.type = SchemaType::array};
        Element default_element{};
        result.items = std::make_shared<Schema>(make_schema(default_element));
        return result;
    }
    else if constexpr (glz::reflectable<Value>) {
        Schema result{
            .type = SchemaType::object,
            .properties = std::map<std::string, std::shared_ptr<Schema>>{},
            .required = std::vector<std::string>{},
            .additional_properties = false,
        };
        const auto& keys = glz::reflect<Value>::keys;
        std::size_t key_index = 0;

        glz::for_each_field(value, [&](auto& member) {
            using Member = std::remove_cvref_t<decltype(member)>;
            const bool required = member_is_required<Member>();
            result.properties->emplace(
                std::string{keys[key_index]},
                std::make_shared<Schema>(make_schema(member)));
            if (required) {
                result.required->emplace_back(keys[key_index]);
            }
            ++key_index;
        });

        return result;
    }
    else {
        static_assert(dependent_false_v<Value>,
                      "cail::schema does not support this C++ type yet");
    }
}

} // namespace detail

template <typename T>
[[nodiscard]] Schema schema()
{
    static_assert(std::default_initializable<T>,
                  "cail::schema<T>() requires a default-initializable model");
    T default_value{};
    return detail::make_schema(default_value);
}

template <typename T>
[[nodiscard]] Result<std::string> json_schema()
{
    return to_json(schema<T>());
}

} // namespace cail

namespace glz {

template <>
struct meta<cail::Schema> {
    using T = cail::Schema;
    static constexpr auto value = object(
        &T::type,
        "description", &T::description,
        "minimum", &T::minimum,
        "maximum", &T::maximum,
        "enum", &T::enum_values,
        "properties", &T::properties,
        "required", &T::required,
        "items", &T::items,
        "additionalProperties", &T::additional_properties);
};

} // namespace glz
