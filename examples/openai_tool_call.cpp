#include <cail/cail.hpp>

#include <iostream>
#include <string>

struct WeatherQuery {
    std::string location;
};

struct WeatherReport {
    std::string location;
    int temperature_fahrenheit{};
};

int main() {
    auto weather_tool = cail::tool<WeatherQuery, WeatherReport>(
        "weather",
        "Get the weather in a location.",
        [](const WeatherQuery& query) {
            // Replace this stub with a real weather service call.
            return WeatherReport{
                .location = query.location,
                .temperature_fahrenheit = 72,
            };
        });

    const auto result = cail::generate_text({
        .model = cail::openai("gpt-6-luna"),
        .prompt = "Use the weather tool to fetch the weather for San Francisco.",
        .tools = {weather_tool},
    });
    if (!result) {
        std::cerr << result.error().message << '\n';
        return 1;
    }
    const auto report = weather_tool.decode_output(result->tool_results.front());
    std::cout << "Tool result: " << report->location << " is "
              << report->temperature_fahrenheit << " F\n";
    std::cout << "Answer: " << result->text << '\n';
}
