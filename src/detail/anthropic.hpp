#pragma once

#include <cail/anthropic.hpp>
#include <glaze/core/common.hpp>

#include <cstddef>
#include <memory>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

namespace cail::detail::anthropic {

struct Config {
  std::string api_key;
  std::string model;
  std::string base_url{"https://api.anthropic.com/v1"};
  std::size_t max_tokens{1024};
  std::vector<HttpHeader> headers;
  std::string request_session_header;
};

struct TextBlock {
  std::string type{"text"};
  std::string text;
};
struct ImageSource {
  std::string type{"base64"};
  std::string media_type;
  std::string data;
};
struct ImageBlock {
  std::string type{"image"};
  ImageSource source;
};
struct ToolUseBlock {
  std::string type{"tool_use"};
  std::string id;
  std::string name;
  glz::raw_json input;
};
struct ToolResultBlock {
  std::string type{"tool_result"};
  std::string tool_use_id;
  glz::raw_json content;
};
struct InputMessage {
  std::string role;
  std::vector<glz::raw_json> content;
};
struct Tool {
  std::string name;
  std::string description;
  glz::raw_json input_schema;
};
struct RequestBody {
  std::string model;
  std::size_t max_tokens{};
  std::vector<InputMessage> messages;
  std::optional<std::string> system;
  std::optional<std::vector<Tool>> tools;
  std::optional<bool> stream;
  struct OutputConfig {
    struct Format {
      std::string type{"json_schema"};
      glz::raw_json schema;
    } format;
  };
  std::optional<OutputConfig> output_config;
  std::optional<double> temperature;
  std::optional<double> top_p;
  std::optional<glz::generic> tool_choice;
  std::optional<std::vector<std::string>> stop_sequences;
};
struct Usage {
  std::optional<std::size_t> input_tokens;
  std::optional<std::size_t> output_tokens;
  std::optional<std::size_t> cache_read_input_tokens;
  std::optional<std::size_t> cache_creation_input_tokens;
};
struct OutputBlock {
  std::string type;
  std::optional<std::string> text;
  std::optional<std::string> thinking;
  std::optional<std::string> signature;
  std::optional<std::string> data;
  std::optional<std::string> id;
  std::optional<std::string> name;
  glz::raw_json input;
};
struct ResponseBody {
  std::vector<OutputBlock> content;
  std::optional<std::string> stop_reason;
  std::optional<Usage> usage;
};
struct ProviderError {
  std::string type;
  std::string message;
};
struct StreamDelta {
  std::string type;
  std::optional<std::string> text;
  std::optional<std::string> thinking;
  std::optional<std::string> signature;
  std::optional<std::string> partial_json;
  std::optional<std::string> stop_reason;
};
struct StreamBody {
  std::string type;
  std::optional<std::size_t> index;
  std::optional<OutputBlock> content_block;
  std::optional<StreamDelta> delta;
  std::optional<ResponseBody> message;
  std::optional<Usage> usage;
  std::optional<ProviderError> error;
};

[[nodiscard]] Result<RequestBody> encode(const GenerationRequest& request, const Config& config,
                                         bool streaming);

void apply_usage(TokenUsage& target, const Usage& usage);

struct ThinkingBlock {
  std::string type;
  std::optional<std::string> thinking;
  std::optional<std::string> signature;
  std::optional<std::string> data;
};

[[nodiscard]] Result<void> retain_thinking(GenerationResponse& result,
                                           const std::vector<ThinkingBlock>& blocks);

[[nodiscard]] Result<GenerationResponse> decode(const ResponseBody& body);

[[nodiscard]] Result<std::string> apply_message_options(std::string_view encoded,
                                                        const GenerationRequest& request);

class Client {
public:
  explicit Client(Config config,
                  std::unique_ptr<HttpTransport> transport = cail::make_default_http_transport());

  [[nodiscard]] Result<GenerationResponse> generate(const GenerationRequest& request,
                                                    std::stop_token stop = {}) const;
  [[nodiscard]] Result<GenerationResponse> stream(const GenerationRequest& request,
                                                  const StreamHandler& on_event,
                                                  std::stop_token stop = {}) const;

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

  [[nodiscard]] Result<GenerationResponse>
  run(const GenerationRequest& request, const StreamHandler& on_event, std::stop_token stop) const;

  Config config_;
  std::unique_ptr<HttpTransport> transport_;
};

} // namespace cail::detail::anthropic
