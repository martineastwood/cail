#include <cail/cail.hpp>

#include <optional>
#include <string>

struct Response {
  cail::Field<double> confidence{.description = "Confidence"};
  std::optional<std::string> explanation;
};

cail::Task<cail::Result<cail::GenerationResponse>> coroutine_smoke() {
  cail::LanguageModel model({}, {}, {}, [](auto, auto complete, auto) -> cail::Result<void> {
    complete(cail::GenerationResponse{.text = "installed"});
    return {};
  });
  co_return co_await cail::generate_text_async({.model = model, .prompt = "hello"});
}

int main() {
  Response response{.confidence = {.value = 0.8}, .explanation = "clear"};
  const auto json = cail::to_json(response);
  if (!json) {
    return 1;
  }
  const auto decoded = cail::from_json<Response>(*json);
  const auto model = cail::openrouter("openai/gpt-4o-mini");
  const auto anthropic = cail::anthropic("claude-haiku-4-5-20251001");
  const auto gemini = cail::gemini("gemini-3.5-flash-lite");
  const auto mistral = cail::mistral("mistral-vibe-cli-with-tools");
  const auto hyper = cail::hyper("deepseek-v4-flash");
  const auto ollama_cloud = cail::ollama_cloud("gemma4:31b");
  const auto awaited = cail::run(coroutine_smoke());
  return awaited && awaited->text == "installed" && decoded && decoded->confidence.value == 0.8 &&
                 decoded->explanation == "clear" &&
                 model.adapter_capabilities().structured_output &&
                 anthropic.adapter_capabilities().tools &&
                 gemini.adapter_capabilities().streaming && mistral.adapter_capabilities().tools &&
                 hyper.adapter_capabilities().streaming && ollama_cloud.adapter_capabilities().tools
             ? 0
             : 1;
}
