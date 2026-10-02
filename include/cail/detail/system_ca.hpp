#pragma once

#include <expected>
#include <system_error>

namespace glz {
struct http_client;
}

namespace cail::detail {

// Call before sending requests. Explicit SSL_CERT_FILE/SSL_CERT_DIR replace system trust.
[[nodiscard]] std::expected<void, std::error_code> configure_system_ca(glz::http_client& client);

} // namespace cail::detail
