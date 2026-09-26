#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
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

[[nodiscard]] inline std::optional<std::string> base64_decode(std::string_view encoded)
{
    constexpr std::string_view alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    if (encoded.size() % 4 != 0) {
        return std::nullopt;
    }
    std::string bytes;
    bytes.reserve((encoded.size() / 4) * 3);
    for (std::size_t i = 0; i < encoded.size(); i += 4) {
        std::uint32_t value = 0;
        unsigned int padding = 0;
        for (std::size_t j = 0; j < 4; ++j) {
            const char character = encoded[i + j];
            if (character == '=') {
                ++padding;
                value <<= 6U;
                continue;
            }
            if (padding != 0) {
                return std::nullopt;
            }
            const auto index = alphabet.find(character);
            if (index == std::string_view::npos) {
                return std::nullopt;
            }
            value = (value << 6U) | static_cast<std::uint32_t>(index);
        }
        bytes.push_back(static_cast<char>((value >> 16U) & 0xffU));
        if (padding < 2U) {
            bytes.push_back(static_cast<char>((value >> 8U) & 0xffU));
        }
        if (padding < 1U) {
            bytes.push_back(static_cast<char>(value & 0xffU));
        }
    }
    return bytes;
}

[[nodiscard]] inline std::string image_data_url(std::string_view mime_type, std::string_view bytes)
{
    return "data:" + std::string{mime_type} + ";base64," + base64_encode(bytes);
}

} // namespace cail::detail
