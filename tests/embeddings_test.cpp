#include "test_support.hpp"

#include <cail/anthropic.hpp>
#include <cail/detail/chat_completions_preset.hpp>
#include <cail/detail/openai_embeddings.hpp>
#include <cail/foundry.hpp>
#include <cail/gemini.hpp>
#include <cail/hyper.hpp>
#include <cail/local.hpp>
#include <cail/mistral.hpp>
#include <cail/ollama_cloud.hpp>
#include <cail/openai.hpp>
#include <cail/opencode.hpp>
#include <cail/openrouter.hpp>

#include <memory>
#include <string>
#include <vector>

namespace test {

[[nodiscard]] inline cail::detail::EmbeddingClientSettings
openai_compatible_settings(std::unique_ptr<cail::HttpTransport> transport) {
  return cail::detail::EmbeddingClientSettings{
      .endpoint = "https://api.openai.com/v1/embeddings",
      .api_key = "test-key",
      .model = "text-embedding-3-small",
      .transport = std::move(transport),
  };
}

void test_openai_compatible_embeddings() {
  auto transport = std::make_unique<StubTransport>();
  auto* stub = transport.get();
  stub->response = cail::HttpResponse{
      .status_code = 200,
      .body = R"({"model":"text-embedding-3-small","data":[)"
              R"({"index":1,"embedding":[0.4,0.5,0.6]},)"
              R"({"index":0,"embedding":[0.1,0.2,0.3]}],)"
              R"("usage":{"prompt_tokens":7}})",
  };
  const cail::detail::EmbeddingClient client(openai_compatible_settings(std::move(transport)));

  const auto batch = client.embed_many({"a red apple", "a green pear"});
  check(batch.has_value(), "OpenAI-compatible embeddings parse a batch");
  check(stub->request.url == "https://api.openai.com/v1/embeddings",
        "OpenAI-compatible embeddings post to the endpoint");
  check(stub->request.headers.front().name == "Authorization" &&
            stub->request.headers.front().value == "Bearer test-key",
        "OpenAI-compatible embeddings send the bearer token");
  check(stub->request.body == R"({"model":"text-embedding-3-small",)"
                              R"("input":["a red apple","a green pear"],)"
                              R"("encoding_format":"float"})",
        "OpenAI-compatible embeddings send the model and inputs");
  if (!batch) {
    return;
  }
  check(batch->model == "text-embedding-3-small" && batch->dimensions == 3 &&
            batch->input_tokens == 7,
        "OpenAI-compatible embeddings report the model, dimensions, and usage");
  check(batch->embeddings.size() == 2 && batch->embeddings[0].values[0] == 0.1F &&
            batch->embeddings[1].values[0] == 0.4F,
        "OpenAI-compatible embeddings restore input order from the returned index");
}

void test_embeddings_dimension_request() {
  auto transport = std::make_unique<StubTransport>();
  auto* stub = transport.get();
  stub->response = cail::HttpResponse{
      .status_code = 200,
      .body = R"({"model":"text-embedding-3-small","data":[{"index":0,"embedding":[0.1,0.2]}]})",
  };
  auto settings = openai_compatible_settings(std::move(transport));
  settings.dimensions = 2;
  const cail::detail::EmbeddingClient client(std::move(settings));

  const auto batch = client.embed_many({"one"});
  check(batch.has_value(), "embeddings accept a requested dimension count");
  check(stub->request.body.find(R"("dimensions":2)") != std::string::npos,
        "embeddings send the requested dimension count");
}

void test_embeddings_without_api_key() {
  auto transport = std::make_unique<StubTransport>();
  auto* stub = transport.get();
  stub->response = cail::HttpResponse{
      .status_code = 200,
      .body = R"({"model":"nomic-embed-text","data":[{"index":0,"embedding":[0.1,0.2]}]})",
  };
  auto settings = openai_compatible_settings(std::move(transport));
  settings.endpoint = "http://127.0.0.1:8080/v1/embeddings";
  settings.api_key.clear();
  const cail::detail::EmbeddingClient client(std::move(settings));

  const auto batch = client.embed_many({"one"});
  check(batch.has_value(), "a self-hosted endpoint needs no API key");
  check(stub->request.headers.size() == 1 && stub->request.headers.front().name == "Content-Type",
        "an endpoint without a key sends no Authorization header");
}

void test_embeddings_validation() {
  auto transport = std::make_unique<StubTransport>();
  auto* stub = transport.get();
  stub->response = cail::HttpResponse{
      .status_code = 200,
      .body = R"({"model":"text-embedding-3-small","data":[{"index":0,"embedding":[0.1]}]})",
  };
  auto settings = openai_compatible_settings(std::move(transport));
  settings.dimensions = 3;
  const cail::detail::EmbeddingClient client(std::move(settings));

  const auto short_vector = client.embed_many({"one"});
  check(!short_vector && short_vector.error().code == cail::ErrorCode::provider_response,
        "embeddings reject a vector that ignores the requested dimensions");

  stub->response.body =
      R"({"model":"text-embedding-3-small","data":[{"index":0,"embedding":[0.1,0.2]}]})";
  const auto missing_vector = client.embed_many({"one", "two"});
  check(!missing_vector && missing_vector.error().code == cail::ErrorCode::provider_response,
        "embeddings reject an incomplete batch");

  stub->response.status_code = 401;
  stub->response.body = R"({"error":{"message":"bad key"}})";
  const auto unauthorized = client.embed_many({"one"});
  check(!unauthorized && unauthorized.error().code == cail::ErrorCode::http_status &&
            unauthorized.error().message == "bad key" && unauthorized.error().http_status == 401,
        "embeddings map provider errors to an HTTP status error");
}

void test_embedding_model_requires_one_vector() {
  cail::EmbeddingModel missing;
  const auto unconfigured = missing.embed("hello");
  check(!unconfigured && unconfigured.error().code == cail::ErrorCode::invalid_configuration,
        "a default embedding model reports that it has no provider");

  cail::EmbeddingModel model{[](const std::vector<std::string>& inputs) {
    return cail::Result<cail::EmbeddingBatch>{cail::EmbeddingBatch{
        .embeddings = std::vector<cail::Embedding>(inputs.size()), .model = "stub"}};
  }};
  check(model.embed("hello").has_value(), "embed returns the single vector for one input");
  check(!model.embed_many({}).has_value(), "embeddings reject an empty batch");
  check(!model.embed_many({""}).has_value(), "embeddings reject an empty input");
}

void test_preset_embeddings_endpoints() {
  check(
      cail::detail::embeddings_endpoint_from("https://api.mistral.ai/v1/chat/completions") ==
              "https://api.mistral.ai/v1/embeddings" &&
          cail::detail::embeddings_endpoint_from("https://openrouter.ai/api/v1/chat/completions") ==
              "https://openrouter.ai/api/v1/embeddings" &&
          cail::detail::embeddings_endpoint_from("http://127.0.0.1:8080/v1/chat/completions") ==
              "http://127.0.0.1:8080/v1/embeddings",
      "presets derive the embeddings endpoint from the chat endpoint");
  check(cail::detail::embeddings_endpoint_from("https://example.com/embeddings") ==
            "https://example.com/embeddings",
        "an embeddings endpoint is left unchanged");

  check(static_cast<bool>(cail::mistral.embedding_model("mistral-embed")) &&
            static_cast<bool>(cail::openrouter.embedding_model("openai/text-embedding-3-small")) &&
            static_cast<bool>(cail::local.embedding_model("nomic-embed-text")) &&
            static_cast<bool>(cail::ollama_cloud.embedding_model("nomic-embed-text")) &&
            static_cast<bool>(cail::hyper.embedding_model("embed-model")) &&
            static_cast<bool>(cail::anthropic.embedding_model("embed-model")) &&
            static_cast<bool>(cail::openai.embedding_model("embed-model")) &&
            static_cast<bool>(cail::gemini.embedding_model("embed-model")) &&
            static_cast<bool>(
                cail::create_opencode({.api_key = "key"}).embedding_model("embed-model")) &&
            static_cast<bool>(cail::foundry.embedding_model("embed-model", "https://foundry.test"
                                                                           "/embeddings")),
        "every provider offers an embedding model");

  auto transport = std::make_unique<StubTransport>();
  auto* stub = transport.get();
  stub->response = cail::HttpResponse{
      .status_code = 200,
      .body = R"({"model":"pin","data":[{"index":0,"embedding":[0.1]}]})",
  };
  cail::MistralSettings settings;
  settings.endpoint = "https://proxy.internal/v1/chat/completions";
  check(cail::MistralSettings{}.endpoint == "https://api.mistral.ai/v1/chat/completions",
        "a preset keeps its provider chat endpoint by default");
  const cail::detail::EmbeddingClient probe(cail::detail::EmbeddingClientSettings{
      .endpoint = cail::detail::embeddings_endpoint_from(settings.endpoint),
      .model = "pin",
      .transport = std::move(transport),
  });
  const auto batch = probe.embed_many({"one"});
  check(batch.has_value() && stub->request.url == "https://proxy.internal/v1/embeddings",
        "an overridden preset endpoint carries over to embeddings");
}

void test_gemini_embeddings() {
  auto transport = std::make_unique<StubTransport>();
  auto* stub = transport.get();
  stub->response = cail::HttpResponse{
      .status_code = 200,
      .body = R"({"embeddings":[{"values":[0.1,0.2,0.3]},{"values":[0.4,0.5,0.6]}],)"
              R"("usageMetadata":{"promptTokenCount":11}})",
  };
  const cail::detail::gemini::EmbeddingClient client(
      cail::detail::gemini::Config{.api_key = "test-key", .model = "gemini-embedding-001"}, 3,
      std::move(transport));

  const auto batch = client.embed_many({"a red apple", "a green pear"});
  check(batch.has_value(), "Gemini embeddings parse a batch");
  check(stub->request.url ==
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001"
            ":batchEmbedContents",
        "Gemini embeddings post to batchEmbedContents without a doubled model prefix");
  check(stub->request.headers.front().name == "x-goog-api-key" &&
            stub->request.headers.front().value == "test-key",
        "Gemini embeddings send the API key header");
  check(stub->request.body ==
            R"({"requests":[{"model":"models/gemini-embedding-001",)"
            R"("content":{"parts":[{"text":"a red apple"}]},"outputDimensionality":3},)"
            R"({"model":"models/gemini-embedding-001",)"
            R"("content":{"parts":[{"text":"a green pear"}]},"outputDimensionality":3}]})",
        "Gemini embeddings send one request per input with the model prefix");
  if (!batch) {
    return;
  }
  check(batch->model == "gemini-embedding-001" && batch->dimensions == 3 &&
            batch->input_tokens == 11,
        "Gemini embeddings report the model, dimensions, and usage");
  check(batch->embeddings.size() == 2 && batch->embeddings[1].values[0] == 0.4F,
        "Gemini embeddings keep input order");

  stub->response.body = R"({"embeddings":[{"values":[0.1]}]})";
  const auto incomplete = client.embed_many({"one", "two"});
  check(!incomplete && incomplete.error().code == cail::ErrorCode::provider_response,
        "Gemini embeddings reject an incomplete batch");

  check(static_cast<bool>(
            cail::create_gemini({.api_key = "test-key"}).embedding_model("gemini-embedding-001")),
        "the Gemini provider offers an embedding model");
}

} // namespace test

int main() {
  test::test_openai_compatible_embeddings();
  test::test_embeddings_dimension_request();
  test::test_embeddings_without_api_key();
  test::test_embeddings_validation();
  test::test_embedding_model_requires_one_vector();
  test::test_preset_embeddings_endpoints();
  test::test_gemini_embeddings();
  return test::failures == 0 ? 0 : 1;
}
