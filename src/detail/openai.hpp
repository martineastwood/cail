#pragma once

#include <cail/openai.hpp>

#include <memory>
#include <stop_token>
#include <string>
#include <string_view>

namespace cail::detail::openai {

struct Config {
  std::string api_key;
  std::string model;
  std::string base_url{"https://api.openai.com/v1"};
  std::string request_session_header;
  bool prompt_cache_key = false;
};

class Client {
public:
  explicit Client(Config config);

  Client(Config config, std::unique_ptr<HttpTransport> transport);

  [[nodiscard]] Result<GenerationResponse> generate(const GenerationRequest& request,
                                                    std::stop_token stop = {}) const;

  [[nodiscard]] Result<GenerationResponse> stream(const GenerationRequest& request,
                                                  const StreamHandler& on_event,
                                                  std::stop_token stop = {}) const;

  [[nodiscard]] Result<GenerationResponse>
  stream(std::string_view prompt, const StreamHandler& on_event, std::stop_token stop = {}) const;

  [[nodiscard]] Result<void> generate_async(GenerationRequest request,
                                            LanguageModel::GenerationCompletion complete,
                                            std::stop_token stop = {}) const;

  [[nodiscard]] Result<void> stream_async(GenerationRequest request, StreamHandler on_event,
                                          LanguageModel::GenerationCompletion complete,
                                          std::stop_token stop = {}) const;

private:
  struct StreamState;

  [[nodiscard]] static Result<GenerationResponse>
  decode_http_response(const HttpResponse& response);

  [[nodiscard]] Result<HttpRequest> make_http_request(const GenerationRequest& request,
                                                      bool streaming) const;

  [[nodiscard]] Result<GenerationResponse> generate_impl(const GenerationRequest& request,
                                                         const StreamHandler& on_event,
                                                         std::stop_token stop = {}) const;

public:
  [[nodiscard]] Result<GenerationResponse> generate(std::string_view prompt) const;

private:
  Config config_;
  std::unique_ptr<HttpTransport> transport_;
};

} // namespace cail::detail::openai

namespace cail::detail {

[[nodiscard]] AdapterCapabilities openai_responses_adapter_capabilities();

[[nodiscard]] std::shared_ptr<openai::Client>
make_openai_responses_client(openai::Config config, std::unique_ptr<HttpTransport> transport = {});

[[nodiscard]] LanguageModel language_model_from(const std::shared_ptr<openai::Client>& client);

} // namespace cail::detail
