#include "detail/embeddings.hpp"
#include "detail/http_context.hpp"
#include <cail/detail/openai_embeddings.hpp>
#include <cail/embedding_model.hpp>
#include <cail/json.hpp>
#include <cail/schema.hpp>
#include <glaze/glaze.hpp>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace cail::detail {

EmbeddingClient::EmbeddingClient(EmbeddingClientSettings settings)
    : settings_(std::move(settings)) {}

Result<EmbeddingBatch> EmbeddingClient::embed_many(const std::vector<std::string>& inputs,
                                                   const std::stop_token& stop) const {
  if (stop.stop_requested()) {
    return std::unexpected(generation_cancelled_error());
  }
  auto http = make_http_request(inputs);
  if (!http) {
    return std::unexpected(http.error());
  }
  auto response = settings_.transport->send(*http, stop);
  if (stop.stop_requested()) {
    return std::unexpected(generation_cancelled_error());
  }
  if (!response) {
    return std::unexpected(response.error());
  }
  return decode_http_response(*response, inputs.size());
}

Result<void> EmbeddingClient::embed_many_async(const std::vector<std::string>& inputs,
                                               EmbeddingModel::BatchCompletion complete,
                                               const std::stop_token& stop) const {
  if (stop.stop_requested()) {
    return std::unexpected(generation_cancelled_error());
  }
  if (!complete) {
    return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                 .message = "Async embeddings require a completion handler."});
  }
  auto http = make_http_request(inputs);
  if (!http) {
    return std::unexpected(http.error());
  }
  settings_.transport->send_async(
      std::move(*http),
      [this, input_count = inputs.size(), complete = std::move(complete),
       stop](Result<HttpResponse> response) {
        if (stop.stop_requested()) {
          complete(std::unexpected(generation_cancelled_error()));
        } else if (!response) {
          complete(std::unexpected(response.error()));
        } else {
          complete(decode_http_response(*response, input_count));
        }
      },
      stop);
  return {};
}

Result<HttpRequest>
EmbeddingClient::make_http_request(const std::vector<std::string>& inputs) const {
  if (settings_.model.empty() || settings_.endpoint.empty() || !settings_.transport ||
      (settings_.dimensions && *settings_.dimensions == 0)) {
    return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                 .message = "Embeddings require a model, endpoint, transport, "
                                            "and positive dimensions."});
  }
  auto encoded = to_json(EmbeddingRequestBody{
      .model = settings_.model, .input = inputs, .dimensions = settings_.dimensions});
  if (!encoded) {
    return std::unexpected(encoded.error());
  }
  std::vector<HttpHeader> headers;
  if (!settings_.api_key.empty()) {
    headers.push_back({.name = "Authorization", .value = "Bearer " + settings_.api_key});
  }
  headers.push_back({.name = "Content-Type", .value = "application/json"});
  return HttpRequest{
      .url = settings_.endpoint, .headers = std::move(headers), .body = std::move(*encoded)};
}

Result<EmbeddingBatch> EmbeddingClient::decode_http_response(const HttpResponse& response,
                                                             std::size_t input_count) const {
  const auto context = [&](Error error) {
    return unexpected_with_http_context<EmbeddingBatch>(std::move(error), response);
  };
  if (is_http_error_status(response.status_code)) {
    return context(http_status_error_from_json_body(response));
  }
  EmbeddingResponseBody body;
  if (const auto error = glz::read<glz::opts{.error_on_unknown_keys = false}>(body, response.body);
      error) {
    return context(Error{.code = ErrorCode::provider_response,
                         .message = glz::format_error(error, response.body)});
  }
  if (body.model.empty() || body.data.size() != input_count) {
    return context(Error{.code = ErrorCode::provider_response,
                         .message = "The embeddings endpoint returned an incomplete batch."});
  }
  EmbeddingBatch result{.embeddings = std::vector<Embedding>(input_count),
                        .model = body.model,
                        .input_tokens =
                            body.usage ? std::optional{body.usage->prompt_tokens} : std::nullopt};
  std::vector<bool> seen(input_count);
  for (auto& item : body.data) {
    if (item.index >= input_count || seen[item.index] || item.embedding.empty() ||
        (result.dimensions && item.embedding.size() != result.dimensions) ||
        (settings_.dimensions && item.embedding.size() != *settings_.dimensions)) {
      return context(Error{.code = ErrorCode::provider_response,
                           .message = "The embeddings endpoint returned an invalid index or vector "
                                      "dimension."});
    }
    seen[item.index] = true;
    result.dimensions = item.embedding.size();
    result.embeddings[item.index] = Embedding{
        .values = std::move(item.embedding), .model = body.model, .dimensions = result.dimensions};
  }
  return result;
}

} // namespace cail::detail

namespace cail::detail {

[[nodiscard]] EmbeddingModel make_embedding_model(EmbeddingClientSettings settings) {
  auto client = std::make_shared<EmbeddingClient>(std::move(settings));
  return EmbeddingModel{
      [client](const std::vector<std::string>& inputs, const std::stop_token& stop) {
        return client->embed_many(inputs, stop);
      },
      [client](const std::vector<std::string>& inputs, EmbeddingModel::BatchCompletion complete,
               const std::stop_token& stop) {
        return client->embed_many_async(
            inputs,
            [client, complete = std::move(complete)](Result<EmbeddingBatch> result) {
              complete(std::move(result));
            },
            stop);
      }};
}

[[nodiscard]] EmbeddingModel
make_openai_style_embedding_model(std::string endpoint, std::string api_key, std::string model,
                                  std::optional<std::size_t> dimensions) {
  return make_embedding_model(EmbeddingClientSettings{
      .endpoint = std::move(endpoint),
      .api_key = std::move(api_key),
      .model = std::move(model),
      .dimensions = dimensions,
  });
}

} // namespace cail::detail
