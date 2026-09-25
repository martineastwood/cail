#pragma once

#include <optional>
#include <string>
#include <utility>

namespace cail {

template <typename T>
struct Field {
    T value{};
    std::string description;
    std::optional<T> minimum;
    std::optional<T> maximum;

    [[nodiscard]] T& get() & noexcept { return value; }
    [[nodiscard]] const T& get() const& noexcept { return value; }

    Field& operator=(T new_value)
    {
        value = std::move(new_value);
        return *this;
    }
};

} // namespace cail
