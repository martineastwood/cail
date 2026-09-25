#pragma once

#include <cail/error.hpp>

#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace cail {

struct HttpHeader {
    std::string name;
    std::string value;
};

struct HttpRequest {
    std::string method{"POST"};
    std::string url;
    std::vector<HttpHeader> headers;
    std::string body;
};

struct HttpResponse {
    int status_code{};
    std::vector<HttpHeader> headers;
    std::string body;
};

using HttpDataHandler = std::function<void(std::string_view)>;

class HttpTransport {
    public:
    virtual ~HttpTransport() = default;

    [[nodiscard]] virtual Result<HttpResponse> send(const HttpRequest& request) = 0;
    // Delivers response body bytes incrementally; the returned body is not buffered.
    [[nodiscard]] virtual Result<HttpResponse> stream(const HttpRequest& request, const HttpDataHandler& on_data) = 0;
};

} // namespace cail
