#pragma once

#include <cail/json.hpp>

#include <glaze/glaze.hpp>

#include <vector>

namespace cail::detail {

template <typename T>
[[nodiscard]] inline Result<void> append_json(std::vector<glz::raw_json>& parts, const T& value)
{
    auto encoded = to_json(value);
    if (!encoded) {
        return std::unexpected(encoded.error());
    }
    parts.emplace_back(std::move(*encoded));
    return {};
}

} // namespace cail::detail
