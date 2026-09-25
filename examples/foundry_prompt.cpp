#include <cail/cail.hpp>

#include <cstdlib>
#include <iostream>

int main() {
  if (!std::getenv("AZURE_FOUNDRY_API_KEY")) {
    std::cerr << "Set AZURE_FOUNDRY_API_KEY before running this example.\n";
    return 1;
  }

  const auto model = cail::create_foundry_model({
      .endpoint =
          "https://<resource>.cognitiveservices.azure.com/openai/responses"
          "?api-version=2025-04-01-preview",
      .deployment = "gpt-5.1-codex",
  });

  const auto response = cail::generate_text({
      .model = model,
      .prompt =
          "Name one advantage of native C++ AI applications in one sentence.",
  });
  if (!response) {
    std::cerr << response.error().message << '\n';
    return 1;
  }
  std::cout << response->text << '\n';
}
