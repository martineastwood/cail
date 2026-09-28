#pragma once

#include <cail/generation.hpp>
#include <cail/http.hpp>

#include <iostream>
#include <memory>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace test {

inline int failures{};

inline void check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

struct Address {
  std::string city;
};

struct Profile {
  std::string answer;
  std::optional<int> confidence;
  std::optional<Address> address;
};

struct ToolInput {
  std::string query;
  std::optional<int> limit;
};

struct ToolOutput {
  int count{};
};

class StubTransport final : public cail::HttpTransport {
public:
  cail::HttpResponse response{
      .status_code = 200,
      .body = R"({"status":"completed","output":[]})",
  };
  std::vector<std::string> chunks;
  cail::HttpRequest request;
  int send_count{};
  int stream_count{};

  [[nodiscard]] cail::Result<cail::HttpResponse> send(const cail::HttpRequest& value) override {
    ++send_count;
    request = value;
    return response;
  }

  [[nodiscard]] cail::Result<cail::HttpResponse> stream(const cail::HttpRequest& value,
                                                        const cail::HttpDataHandler& on_data,
                                                        std::stop_token stop) override {
    ++stream_count;
    request = value;
    for (const auto& chunk : chunks) {
      if (stop.stop_requested()) {
        return std::unexpected(
            cail::Error{.code = cail::ErrorCode::cancelled, .message = "Cancelled."});
      }
      on_data(chunk);
    }
    if (stop.stop_requested()) {
      return std::unexpected(
          cail::Error{.code = cail::ErrorCode::cancelled, .message = "Cancelled."});
    }
    return response;
  }
};

class ScriptedClient {
public:
  explicit ScriptedClient(std::vector<cail::GenerationResponse> responses)
      : responses_(std::move(responses)) {}

  [[nodiscard]] cail::Result<cail::GenerationResponse>
  generate(const cail::GenerationRequest& request) const {
    requests.push_back(request);
    if (responses_.empty()) {
      return std::unexpected(cail::Error{
          .code = cail::ErrorCode::provider_response,
          .message = "The test client ran out of scripted responses.",
      });
    }
    auto response = std::move(responses_.front());
    responses_.erase(responses_.begin());
    return response;
  }

  [[nodiscard]] cail::Result<cail::GenerationResponse>
  stream(const cail::GenerationRequest& request, const cail::StreamHandler& on_event,
         std::stop_token stop) const {
    if (stop.stop_requested()) {
      return std::unexpected(cail::generation_cancelled_error());
    }
    auto response = generate(request);
    if (response && !response->text.empty()) {
      on_event(cail::TextDelta{.text = response->text});
    }
    return response;
  }

  mutable std::vector<cail::GenerationRequest> requests;

private:
  mutable std::vector<cail::GenerationResponse> responses_;
};

} // namespace test
