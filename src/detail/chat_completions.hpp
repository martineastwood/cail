#pragma once

#include <cail/chat_completions.hpp>
#include <glaze/core/common.hpp>

#include <cstddef>
#include <memory>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

namespace cail::detail::chat_completions {

struct TextPart {
  std::string type{"text"};
  std::string text;
};
struct ImageUrl {
  std::string url;
};
struct ImagePart {
  std::string type{"image_url"};
  ImageUrl image_url;
};
struct FunctionCall {
  std::string name;
  std::string arguments;
};
struct ToolCall {
  std::optional<std::size_t> index;
  std::string id;
  std::string type{"function"};
  FunctionCall function;
};
struct ToolDefinition {
  std::string type{"function"};
  struct Function {
    std::string name;
    std::string description;
    glz::raw_json parameters;
  } function;
};
struct InputMessage {
  std::string role;
  std::optional<glz::raw_json> content;
  std::optional<std::string> tool_call_id;
  std::optional<std::vector<ToolCall>> tool_calls;
};
struct RequestBody {
  std::string model;
  std::vector<glz::raw_json> messages;
  std::optional<std::vector<glz::raw_json>> tools;
  std::optional<bool> stream;
  struct StreamOptions {
    bool include_usage{true};
  };
  std::optional<StreamOptions> stream_options;
  struct JsonSchema {
    std::string name;
    std::optional<std::string> description;
    bool strict{true};
    glz::raw_json schema;
  };
  struct ResponseFormat {
    std::string type{"json_schema"};
    JsonSchema json_schema;
  };
  std::optional<ResponseFormat> response_format;
  std::optional<std::size_t> max_tokens;
  std::optional<double> temperature;
  std::optional<double> top_p;
  std::optional<glz::generic> tool_choice;
  std::optional<std::vector<std::string>> stop;
};
struct Usage {
  std::size_t prompt_tokens{};
  std::size_t completion_tokens{};
  struct Details {
    std::optional<std::size_t> cached_tokens;
    std::optional<std::size_t> reasoning_tokens;
  };
  std::optional<Details> prompt_tokens_details;
  std::optional<Details> completion_tokens_details;
};
struct ProviderError {
  std::string message;
  std::optional<std::string> code;
  std::optional<std::string> type;
};
struct OutputToolCall {
  std::optional<std::size_t> index;
  std::optional<std::string> id;
  struct Function {
    std::optional<std::string> name;
    std::optional<std::string> arguments;
  };
  std::optional<Function> function;
};
struct OutputMessage {
  std::optional<glz::raw_json> content;
  std::optional<std::string> refusal;
  std::optional<std::string> reasoning_content;
  std::optional<glz::raw_json> reasoning_details;
  std::optional<std::vector<OutputToolCall>> tool_calls;
};
struct Choice {
  std::size_t index{};
  std::optional<std::string> finish_reason;
  std::optional<OutputMessage> message;
  std::optional<OutputMessage> delta;
};
struct ResponseBody {
  std::vector<Choice> choices;
  std::optional<Usage> usage;
  std::optional<ProviderError> error;
};

struct ContentText {
  std::string text;
  std::string reasoning;
};

void append_content(const glz::generic& value, ContentText& output, bool thinking = false);

[[nodiscard]] Result<ContentText> decode_content(const glz::raw_json& raw);

[[nodiscard]] TokenUsage usage(const Usage& value);

[[nodiscard]] ProviderOptions
round_trip_options(bool retain_reasoning_content, std::string_view reasoning_content,
                   const std::optional<std::string>& reasoning_details_json);

[[nodiscard]] Result<GenerationResponse> decode(const ResponseBody& body,
                                                bool retain_reasoning_content);

[[nodiscard]] Result<RequestBody> encode(const GenerationRequest& request, std::string model,
                                         bool streaming, bool retain_reasoning_content);

class Client {
public:
  Client(cail::ChatCompletionsSettings settings, std::string model,
         std::unique_ptr<HttpTransport> transport = cail::make_default_http_transport());

  [[nodiscard]] Result<GenerationResponse> generate(const GenerationRequest& request,
                                                    const std::stop_token& stop = {}) const;
  [[nodiscard]] Result<void> generate_async(GenerationRequest request,
                                            LanguageModel::GenerationCompletion complete,
                                            const std::stop_token& stop = {}) const;
  [[nodiscard]] Result<GenerationResponse> stream(const GenerationRequest& request,
                                                  const StreamHandler& on_event,
                                                  const std::stop_token& stop = {}) const;

  [[nodiscard]] Result<void> stream_async(GenerationRequest request, StreamHandler on_event,
                                          LanguageModel::GenerationCompletion complete,
                                          const std::stop_token& stop = {}) const;

private:
  struct StreamState;

  [[nodiscard]] Result<HttpRequest> make_http_request(const GenerationRequest& request,
                                                      bool streaming) const;

  [[nodiscard]] static Result<GenerationResponse> decode_http_response(const HttpResponse& response,
                                                                       bool retain_reasoning);

  [[nodiscard]] Result<GenerationResponse> run(const GenerationRequest& request,
                                               const StreamHandler& on_event,
                                               const std::stop_token& stop) const;

  cail::ChatCompletionsSettings settings_;
  std::string model_;
  std::unique_ptr<HttpTransport> transport_;
};

} // namespace cail::detail::chat_completions
