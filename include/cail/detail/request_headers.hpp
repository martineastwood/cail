#pragma once

#include <cail/http.hpp>

#include <string>
#include <string_view>
#include <vector>

namespace cail::detail {

inline void append_session_header(std::vector<HttpHeader>& headers,
                                  std::string_view header_name,
                                  std::string_view session_id)
{
    if (!header_name.empty() && !session_id.empty()) {
        headers.push_back(HttpHeader{
            .name = std::string{header_name},
            .value = std::string{session_id},
        });
    }
}

} // namespace cail::detail
