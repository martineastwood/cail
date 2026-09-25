#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace cail::detail {

[[nodiscard]] inline std::string base64_encode(std::string_view bytes)
{
    constexpr std::string_view alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string result;
    result.reserve(((bytes.size() + 2) / 3) * 4);
    for (std::size_t index = 0; index < bytes.size(); index += 3) {
        const auto first = static_cast<unsigned char>(bytes[index]);
        const auto second = index + 1 < bytes.size() ? static_cast<unsigned char>(bytes[index + 1]) : 0;
        const auto third = index + 2 < bytes.size() ? static_cast<unsigned char>(bytes[index + 2]) : 0;
        result.push_back(alphabet[first >> 2]);
        result.push_back(alphabet[((first & 3U) << 4U) | (second >> 4U)]);
        result.push_back(index + 1 < bytes.size() ? alphabet[((second & 15U) << 2U) | (third >> 6U)] : '=');
        result.push_back(index + 2 < bytes.size() ? alphabet[third & 63U] : '=');
    }
    return result;
}

} // namespace cail::detail
