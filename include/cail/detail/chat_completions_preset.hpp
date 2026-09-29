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
[[nodiscard]] inline Result<std::string> embeddings_endpoint_from(std::string_view chat_endpoint) {
  constexpr std::string_view chat_suffix{"/chat/completions"};
  if (chat_endpoint.ends_with(chat_suffix)) {
    return std::string{chat_endpoint.substr(0, chat_endpoint.size() - chat_suffix.size())} +
           "/embeddings";
  }
  constexpr std::string_view embeddings_suffix{"/embeddings"};
  if (chat_endpoint.ends_with(embeddings_suffix)) {
    return std::string{chat_endpoint};
  }
  return std::unexpected(Error{
      .code = ErrorCode::invalid_configuration,
      .message = "The provider endpoint must end with /chat/completions or /embeddings.",
  });
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
    auto endpoint = embeddings_endpoint_from(settings_.endpoint);
    if (!endpoint) {
      const auto error = endpoint.error();
      return EmbeddingModel{[error](const std::vector<std::string>&) -> Result<EmbeddingBatch> {
        return std::unexpected(error);
      }};
    }
    return make_embedding_model(EmbeddingClientSettings{
        .endpoint = std::move(*endpoint),
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
