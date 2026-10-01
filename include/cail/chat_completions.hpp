#pragma once

#include <cail/http.hpp>
#include <cail/language_model.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace cail {

struct ChatCompletionsSettings {
  std::string endpoint{"https://api.openai.com/v1/chat/completions"};
  std::string api_key;
  std::vector<HttpHeader> headers;
  std::string request_session_header;
  bool prompt_cache_key = false;
  bool session_body = false;
  bool retain_reasoning_content = true;
};

[[nodiscard]] constexpr AdapterCapabilities chat_completions_adapter_capabilities() {
  return AdapterCapabilities{
      .image_input = true,
      .tools = true,
      .structured_output = true,
      .reasoning = true,
  };
}

class ChatCompletionsProvider {
public:
  explicit ChatCompletionsProvider(ChatCompletionsSettings settings)
      : settings_(std::move(settings)) {}
  [[nodiscard]] LanguageModel operator()(std::string model_id) const;

  [[nodiscard]] LanguageModel operator()(std::string model_id,
                                         std::unique_ptr<HttpTransport> transport) const;

private:
  ChatCompletionsSettings settings_;
};

[[nodiscard]] inline ChatCompletionsProvider
create_chat_completions(ChatCompletionsSettings settings) {
  return ChatCompletionsProvider{std::move(settings)};
}

} // namespace cail
