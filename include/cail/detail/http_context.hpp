#pragma once

#include <cail/error.hpp>
#include <cail/http.hpp>

#include <cctype>
#include <string>

namespace cail::detail {

inline void attach_http_context(Error& error, const HttpResponse& response)
{
    error.http_status = response.status_code;
    for (const auto& header : response.headers) {
        std::string name = header.name;
        for (auto& character : name) {
            character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
        }
        if (name == "x-request-id" || name == "request-id") {
            error.request_id = header.value;
            break;
        }
    }
}

template <typename T>
[[nodiscard]] Result<T> unexpected_with_http_context(Error error, const HttpResponse& response)
{
    attach_http_context(error, response);
    return std::unexpected(std::move(error));
}

} // namespace cail::detail
