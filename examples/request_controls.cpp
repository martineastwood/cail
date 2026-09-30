#include <cail/cail.hpp>

#include <iostream>

int main() {
  const auto response = cail::generate_text({
      .model = cail::gemini("gemini-2.5-flash"),
      .prompt = "Give me three concise tips for learning C++.",
      .max_output_tokens = 512,
      .temperature = 0.4,
      .stop_sequences = {"END"},
      .tool_choice = cail::ToolChoice{.mode = cail::ToolChoiceMode::none},
  });
  if (!response) {
    std::cerr << response.error().message << '\n';
    return 1;
  }
  std::cout << response->text << '\n';
  if (response->finish_reason == cail::FinishReason::length) {
    std::cerr << "The model reached its output limit.\n";
  }
  for (const auto& step : response->steps) {
    if (step.usage) {
      std::cout << "Step " << step.step << ": " << step.usage->input_tokens << " input, "
                << step.usage->output_tokens << " output tokens\n";
    }
  }
  if (response->total_usage) {
    std::cout << "Reported total: " << response->total_usage->input_tokens << " input, "
              << response->total_usage->output_tokens << " output tokens\n";
  }
}
