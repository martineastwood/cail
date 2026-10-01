#include <cail/cail.hpp>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

struct WeatherQuery {
  std::string location;
};
struct WeatherReport {
  std::string location;
  int temperature_fahrenheit{};
};

int main() {
  auto weather = cail::tool<WeatherQuery, WeatherReport>(
      "weather", "Fetch weather for a location.", [](const WeatherQuery& query) {
        return WeatherReport{query.location, 72}; // Replace with your weather service.
      });
  cail::Agent agent({
      .model = cail::openai("gpt-6-luna"),
      .instructions = "Help with travel plans. If a tool result contains an error, explain it "
                      "without inventing weather.",
      .tools = {weather},
      .memory = std::make_shared<cail::FileConversationMemory>("conversations"),
      .conversation_id = "traveler-42",
  });
  auto result = agent.generate("Use the weather tool to fetch the weather for Paris.",
                               {.pause_when = [](const cail::ToolCall&) { return true; }});
  while (result && result->tool_continuation) {
    std::vector<cail::ToolResult> outputs;
    for (const auto& call : result->tool_calls) {
      std::cout << call.name << " " << call.arguments << "\nApprove? [y/N] ";
      std::string answer;
      std::getline(std::cin, answer);
      cail::Result<std::string> output = std::string{R"({"error":"Approval denied"})"};
      if (answer == "y") {
        output = weather.execute(call, {});
      }
      if (!output) {
        std::cerr << output.error().message << '\n';
        return 1;
      }
      outputs.push_back({call.id, call.name, *output});
    }
    result = agent.resume(*result->tool_continuation, outputs);
  }
  if (!result) {
    std::cerr << result.error().message << '\n';
    return 1;
  }
  std::cout << result->text << '\n';
  auto follow_up = agent.generate("What city did I ask about?");
  if (!follow_up) {
    std::cerr << follow_up.error().message << '\n';
    return 1;
  }
  std::cout << follow_up->text << '\n';
}
