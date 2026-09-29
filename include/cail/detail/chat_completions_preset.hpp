#pragma once

#include <cail/chat_completions.hpp>
#include <cail/detail/env.hpp>
#include <cail/detail/openai_embeddings.hpp>
#include <cail/embedding_model.hpp>

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cail {

template <typename Tag> struct ChatCompletionsPresetSettings {
  std::string api_key;
  std::vector<HttpHeader> headers;
  std::string endpoint{Tag::endpoint};
  std::string request_session_header;
  bool prompt_cache_key = false;
  bool session_body = false;
  bool retain_reasoning_content = true;
};

namespace detail {

// Presets point at a chat endpoint, so embeddings live at the sibling path:
// "https://api.mistral.ai/v1/chat/completions" -> "https://api.mistral.ai/v1/embeddings".
[[nodiscard]] inline std::string embeddings_endpoint_from(std::string_view chat_endpoint) {
  constexpr std::string_view suffix{"/chat/completions"};
  if (!chat_endpoint.ends_with(suffix)) {
    return std::string{chat_endpoint};
  }
  return std::string{chat_endpoint.substr(0, chat_endpoint.size() - suffix.size())} + "/embeddings";
}

template <typename Tag> class ChatCompletionsPresetProvider {
public:
  explicit ChatCompletionsPresetProvider(ChatCompletionsPresetSettings<Tag> settings = {})
      : settings_(std::move(settings)) {}

  [[nodiscard]] LanguageModel operator()(std::string model_id) const {
    return (*this)(std::move(model_id), make_default_http_transport());
  }

  [[nodiscard]] LanguageModel operator()(std::string model_id,
                                         std::unique_ptr<HttpTransport> transport) const {
    return ChatCompletionsProvider{ChatCompletionsSettings{
        .endpoint = settings_.endpoint,
        .api_key = env_or(settings_.api_key, Tag::env_var),
        .headers = settings_.headers,
        .request_session_header = settings_.request_session_header,
        .prompt_cache_key = settings_.prompt_cache_key,
        .session_body = settings_.session_body,
        .retain_reasoning_content = settings_.retain_reasoning_content,
    }}(std::move(model_id), std::move(transport));
  }

  [[nodiscard]] EmbeddingModel
  embedding_model(std::string model_id,
                  std::optional<std::size_t> dimensions = std::nullopt) const {
    return make_embedding_model(EmbeddingClientSettings{
        .endpoint = embeddings_endpoint_from(settings_.endpoint),
        .api_key = env_or(settings_.api_key, Tag::env_var),
        .model = std::move(model_id),
        .dimensions = dimensions,
    });
  }

private:
  ChatCompletionsPresetSettings<Tag> settings_;
};

} // namespace detail

} // namespace cail
