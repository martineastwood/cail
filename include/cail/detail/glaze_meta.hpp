#pragma once

#include <cail/field.hpp>

#include <glaze/glaze.hpp>
#include <magic_enum/magic_enum.hpp>

#include <type_traits>

namespace glz {

template <typename T>
struct meta<cail::Field<T>> {
    static constexpr auto read = [](cail::Field<T>& field, const T& value) {
        field.value = value;
    };

    static constexpr auto write = [](const cail::Field<T>& field) -> const T& {
        return field.value;
    };

    static constexpr auto value = custom<read, write>;
};

template <typename T>
    requires std::is_enum_v<T>
struct meta<T> {
    static constexpr auto keys = magic_enum::enum_names<T>();
    static constexpr auto value = magic_enum::enum_values<T>();
};

} // namespace glz
