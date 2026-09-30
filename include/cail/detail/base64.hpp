#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace cail::detail {

[[nodiscard]] inline std::string base64_encode(std::string_view bytes) {
  constexpr std::string_view alphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string result;
  result.reserve(((bytes.size() + 2) / 3) * 4);
  for (std::size_t index = 0; index < bytes.size(); index += 3) {
    const auto first = static_cast<unsigned char>(bytes[index]);
    const auto second = index + 1 < bytes.size() ? static_cast<unsigned char>(bytes[index + 1]) : 0;
    const auto third = index + 2 < bytes.size() ? static_cast<unsigned char>(bytes[index + 2]) : 0;
    result.push_back(alphabet[first >> 2]);
    result.push_back(alphabet[((first & 3U) << 4U) | (second >> 4U)]);
    result.push_back(index + 1 < bytes.size() ? alphabet[((second & 15U) << 2U) | (third >> 6U)]
                                              : '=');
    result.push_back(index + 2 < bytes.size() ? alphabet[third & 63U] : '=');
  }
  return result;
}

[[nodiscard]] inline std::optional<std::string> base64_decode(std::string_view encoded) {
  constexpr std::string_view alphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  if (encoded.size() % 4 != 0)
    return std::nullopt;
  std::string bytes;
  bytes.reserve(encoded.size() / 4 * 3);
  for (std::size_t index = 0; index < encoded.size(); index += 4) {
    const auto first = alphabet.find(encoded[index]);
    const auto second = alphabet.find(encoded[index + 1]);
    const bool third_padding = encoded[index + 2] == '=';
    const bool fourth_padding = encoded[index + 3] == '=';
    const auto third = third_padding ? 0 : alphabet.find(encoded[index + 2]);
    const auto fourth = fourth_padding ? 0 : alphabet.find(encoded[index + 3]);
    if (first == std::string_view::npos || second == std::string_view::npos ||
        third == std::string_view::npos || fourth == std::string_view::npos ||
        (third_padding && !fourth_padding) ||
        ((third_padding || fourth_padding) && index + 4 != encoded.size()) ||
        (third_padding && (second & 15U)) || (!third_padding && fourth_padding && (third & 3U)))
      return std::nullopt;
    bytes.push_back(static_cast<char>((first << 2U) | (second >> 4U)));
    if (!third_padding)
      bytes.push_back(static_cast<char>((second << 4U) | (third >> 2U)));
    if (!fourth_padding)
      bytes.push_back(static_cast<char>((third << 6U) | fourth));
  }
  return bytes;
}

[[nodiscard]] inline std::string image_data_url(std::string_view mime_type,
                                                std::string_view bytes) {
  return "data:" + std::string{mime_type} + ";base64," + base64_encode(bytes);
}

} // namespace cail::detail
