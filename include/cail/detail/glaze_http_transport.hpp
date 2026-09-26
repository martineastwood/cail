#pragma once

#include <cail/http.hpp>

#include <glaze/net/http_client.hpp>

#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string>
#include <system_error>
#include <utility>

namespace cail::detail {

class GlazeHttpTransport final : public HttpTransport {
    public:
    [[nodiscard]] Result<HttpResponse> send(const HttpRequest& request) override {
        glz::http_headers headers;
        for (const auto& header : request.headers) {
            headers.add(header.name, header.value);
        }

        std::expected<glz::response, std::error_code> response = [&] {
            if (request.method == "POST") {
                return client_.post(request.url, request.body, headers);
            }
            return std::expected<glz::response, std::error_code>{
                std::unexpected(std::make_error_code(std::errc::operation_not_supported))};
        }();

        if (!response) {
            return std::unexpected(Error{
                .code = ErrorCode::transport,
                .message = response.error().message(),
            });
        }

        HttpResponse result{
            .status_code = response->status_code,
            .body = std::move(response->response_body),
        };
        result.headers.reserve(response->response_headers.size());
        for (const auto& [name, value] : response->response_headers) {
            result.headers.push_back(HttpHeader{.name = name, .value = value});
        }
        return result;
    }

    [[nodiscard]] Result<HttpResponse> stream(
        const HttpRequest& request, const HttpDataHandler& on_data, std::stop_token stop) override {
        if (stop.stop_requested()) {
            return std::unexpected(Error{.code = ErrorCode::cancelled, .message = "The HTTP stream was cancelled."});
        }
        struct State {
            HttpResponse response;
            std::optional<std::error_code> error;
            std::exception_ptr callback_error;
            std::promise<void> disconnected;
            std::once_flag finish;
        };

        auto state = std::make_shared<State>();
        auto done = state->disconnected.get_future();
        glz::http_headers headers;
        for (const auto& header : request.headers) {
            headers.add(header.name, header.value);
        }

        glz::stream_request_params_v2 params{
            .method = request.method,
            .url = request.url,
            .strategy = glz::stream_read_strategy::immediate_delivery,
            .body = request.body,
            .headers = std::move(headers),
            .on_data = [state, on_data, stop](std::string_view bytes) {
                if (state->response.status_code >= 400) {
                    state->response.body.append(bytes);
                    return;
                }
                if (!stop.stop_requested() && !state->callback_error && on_data) {
                    try {
                        on_data(bytes);
                    } catch (...) {
                        state->callback_error = std::current_exception();
                    }
                }
            },
            .on_error = [state](std::error_code error) {
                if (state->response.status_code < 400) {
                    state->error = error;
                }
            },
            .on_progress = [state, stop](std::size_t, std::size_t) {
                return !stop.stop_requested() && !state->callback_error;
            },
            .on_connect = [state](const glz::response& response) {
                state->response.status_code = response.status_code;
                state->response.headers.reserve(response.response_headers.size());
                for (const auto& [name, value] : response.response_headers) {
                    state->response.headers.push_back(HttpHeader{.name = name, .value = value});
                }
            },
            .on_disconnect = [state] { std::call_once(state->finish, [state] { state->disconnected.set_value(); }); },
            .status_is_error = [](int) { return false; },
        };

        auto connection = client_.stream_request_v2(params);
        if (!connection) {
            return std::unexpected(Error{
                .code = ErrorCode::transport,
                .message = "Glaze could not start the HTTP stream.",
            });
        }
        std::stop_callback cancel_on_stop(stop, [connection] { connection->disconnect(); });
        done.wait();
        if (stop.stop_requested()) {
            return std::unexpected(Error{.code = ErrorCode::cancelled, .message = "The HTTP stream was cancelled."});
        }
        if (state->callback_error) {
            return std::unexpected(Error{
                .code = ErrorCode::transport,
                .message = "The HTTP stream callback threw an exception.",
            });
        }
        if (state->error) {
            return std::unexpected(Error{
                .code = ErrorCode::transport,
                .message = state->error->message(),
            });
        }
        return std::move(state->response);
    }

    private:
    glz::http_client client_;
};

} // namespace cail::detail
