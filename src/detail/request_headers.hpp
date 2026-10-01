#pragma once

#include <cail/http.hpp>

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>
#include <vector>

namespace cail::detail {

[[nodiscard]] inline bool has_header(const std::vector<HttpHeader>& headers,
                                     std::string_view name) {
  return std::ranges::any_of(headers, [name](const HttpHeader& header) {
    return std::ranges::equal(header.name, name, [](char left, char right) {
      return std::tolower(static_cast<unsigned char>(left)) ==
             std::tolower(static_cast<unsigned char>(right));
    });
  });
}

inline void append_session_header(std::vector<HttpHeader>& headers, std::string_view header_name,
                                  std::string_view session_id) {
  if (!header_name.empty() && !session_id.empty()) {
    headers.push_back(HttpHeader{
        .name = std::string{header_name},
        .value = std::string{session_id},
    });
  }
}

} // namespace cail::detail
