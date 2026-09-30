#pragma once

#include <cail/generation.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace cail {

// Limit source bytes before provider encoding and request overhead.
inline constexpr std::size_t default_file_limit = 32 * 1024 * 1024;

[[nodiscard]] inline Result<std::string> load_file(const std::filesystem::path& path,
                                                   std::size_t max_bytes = default_file_limit) {
  std::error_code error;
  if (!std::filesystem::is_regular_file(path, error)) {
    return std::unexpected(
        Error{.code = ErrorCode::file, .message = "Not a readable regular file: " + path.string()});
  }
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    return std::unexpected(
        Error{.code = ErrorCode::file, .message = "Cannot open file: " + path.string()});
  }
  const auto size = file.tellg();
  if (size < 0 || static_cast<std::uintmax_t>(size) > max_bytes) {
    return std::unexpected(
        Error{.code = ErrorCode::file,
              .message = "Cannot determine file size or file exceeds max_bytes: " + path.string()});
  }
  std::string bytes(static_cast<std::size_t>(size), '\0');
  file.seekg(0);
  if (!file.read(bytes.data(), static_cast<std::streamsize>(bytes.size()))) {
    return std::unexpected(
        Error{.code = ErrorCode::file, .message = "Cannot read file: " + path.string()});
  }
  return bytes;
}

[[nodiscard]] inline Result<TextPart> load_text(const std::filesystem::path& path,
                                                std::size_t max_bytes = default_file_limit) {
  auto bytes = load_file(path, max_bytes);
  if (!bytes)
    return std::unexpected(bytes.error());
  return TextPart{.text = std::move(*bytes)};
}

[[nodiscard]] inline Result<ImagePart> load_image(const std::filesystem::path& path,
                                                  std::size_t max_bytes = default_file_limit) {
  auto bytes = load_file(path, max_bytes);
  if (!bytes)
    return std::unexpected(bytes.error());
  const std::string_view data = *bytes;
  std::string mime_type;
  if (data.starts_with(std::string_view{"\x89PNG\r\n\x1a\n", 8}))
    mime_type = "image/png";
  else if (data.starts_with(std::string_view{"\xff\xd8\xff", 3}))
    mime_type = "image/jpeg";
  else if (data.starts_with("GIF87a") || data.starts_with("GIF89a"))
    mime_type = "image/gif";
  else if (data.size() >= 12 && data.starts_with("RIFF") && data.substr(8, 4) == "WEBP")
    mime_type = "image/webp";
  else
    return std::unexpected(
        Error{.code = ErrorCode::file,
              .message = "Expected a PNG, JPEG, GIF, or WebP image: " + path.string()});
  return ImagePart{.bytes = std::move(*bytes), .mime_type = std::move(mime_type)};
}

[[nodiscard]] inline Result<PdfPart> load_pdf(const std::filesystem::path& path,
                                              std::size_t max_bytes = default_file_limit) {
  auto bytes = load_file(path, max_bytes);
  if (!bytes)
    return std::unexpected(bytes.error());
  if (!bytes->starts_with("%PDF-")) {
    return std::unexpected(
        Error{.code = ErrorCode::file, .message = "Expected a PDF file: " + path.string()});
  }
  return PdfPart{.bytes = std::move(*bytes), .filename = path.filename().string()};
}

} // namespace cail
