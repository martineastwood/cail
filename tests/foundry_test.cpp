#include "test_support.hpp"

#include <cail/foundry.hpp>

#include <cstdlib>

namespace test {

void test_foundry_provider() {
  auto transport = std::make_unique<StubTransport>();
  auto *stub = transport.get();
  const cail::FoundryProvider provider = cail::create_foundry({
      .api_key = "test-foundry-key",
  });
  const auto model = provider(cail::FoundryDeployment{
      .endpoint = "https://example.test/openai/"
                  "responses?api-version=2025-04-01-preview",
      .deployment = "deployment-a",
  }, std::move(transport));
  const auto response = model.generate(cail::GenerationRequest{
      .messages = {cail::Message{
          .role = cail::MessageRole::user,
          .content = {cail::TextPart{.text = "Reply with exactly OK."}},
      }},
  });
  check(response.has_value(), "Foundry returns a generation response");
  check(stub->request.url == "https://example.test/openai/"
                             "responses?api-version=2025-04-01-preview",
        "Foundry keeps the deployment Responses URL without appending a path");
  check(stub->request.headers.size() >= 2 &&
            stub->request.headers[0].value == "Bearer test-foundry-key",
        "Foundry authenticates with a Bearer API key");
  check(stub->request.body.find("\"model\":\"deployment-a\"") !=
                std::string::npos &&
            stub->request.body.find("Reply with exactly OK.") !=
                std::string::npos,
        "Foundry requests the deployment model with the prompt");
}

void test_foundry_env_fallback() {
  const char *previous = std::getenv("AZURE_FOUNDRY_API_KEY");
  const std::string saved = previous == nullptr ? "" : previous;
  setenv("AZURE_FOUNDRY_API_KEY", "env-foundry-key", 1);
  auto transport = std::make_unique<StubTransport>();
  auto *stub = transport.get();
  const auto model = cail::create_foundry()(
      cail::FoundryDeployment{
          .endpoint = "https://example.test/openai/responses?api-version=1",
          .deployment = "deployment-b",
      },
      std::move(transport));
  const auto response = model.generate(cail::GenerationRequest{
      .messages = {cail::Message{
          .role = cail::MessageRole::user,
          .content = {cail::TextPart{.text = "hello"}},
      }},
  });
  if (!saved.empty()) {
    setenv("AZURE_FOUNDRY_API_KEY", saved.c_str(), 1);
  } else {
    unsetenv("AZURE_FOUNDRY_API_KEY");
  }
  check(response.has_value(), "Foundry reads the API key from the environment");
  check(stub->request.headers.size() >= 1 &&
            stub->request.headers[0].value == "Bearer env-foundry-key",
        "Foundry falls back to AZURE_FOUNDRY_API_KEY");
}

void test_foundry_missing_endpoint() {
  const auto model = cail::create_foundry_model({
      .api_key = "test-foundry-key",
      .endpoint = "",
      .deployment = "deployment-a",
  });
  const auto response = model.generate(cail::GenerationRequest{
      .messages = {cail::Message{
          .role = cail::MessageRole::user,
          .content = {cail::TextPart{.text = "hello"}},
      }},
  });
  check(!response, "Foundry rejects an empty deployment endpoint");
  check(!response &&
            response.error().code == cail::ErrorCode::invalid_configuration,
        "Foundry maps missing endpoint to invalid configuration");
}

void test_foundry_stream() {
  auto transport = std::make_unique<StubTransport>();
  auto *stub = transport.get();
  stub->chunks = {
      "event: response.output_text.delta\ndata: "
      "{\"type\":\"response.output_text.delta\",\"delta\":\"Hi\"}\n\n",
      "event: response.completed\ndata: "
      "{\"type\":\"response.completed\",\"response\":{\"status\":\"completed\","
      "\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\","
      "\"text\":\"Hi\"}]}]}}\n\n",
  };
  const auto model = cail::create_foundry({
      .api_key = "test-foundry-key",
  })(cail::FoundryDeployment{
      .endpoint = "https://example.test/openai/responses?api-version=1",
      .deployment = "deployment-a",
  }, std::move(transport));
  std::string streamed;
  const auto response = model.stream(
      "hello",
      [&](const cail::StreamEvent &event) {
        if (const auto *delta = std::get_if<cail::TextDelta>(&event)) {
          streamed += delta->text;
        }
      });
  check(response.has_value(), "Foundry streams a generation response");
  check(streamed == "Hi", "Foundry forwards streamed text deltas");
  check(stub->request.headers.size() >= 3 &&
            stub->request.headers[2].value == "text/event-stream",
        "Foundry requests an SSE stream");
}

} // namespace test

int main() {
  test::test_foundry_provider();
  test::test_foundry_env_fallback();
  test::test_foundry_missing_endpoint();
  test::test_foundry_stream();
  return test::failures == 0 ? 0 : 1;
}
