#include <cail/cail.hpp>

#include <iostream>
#include <string>
#include <string_view>

int main()
{
    const auto model = cail::openai("gpt-6-luna");
    constexpr std::string_view paragraph =
        "Streaming APIs let applications show partial model output as it is generated, rather than "
        "waiting for the full response. This improves perceived latency for interactive tools such "
        "as chat assistants and code editors. In C++, streaming fits naturally with callbacks that "
        "receive string chunks as they arrive from the network.";
    const std::string prompt =
        std::string("Summarize the main idea of this paragraph:\n\n") + std::string(paragraph);
    auto response = model.stream(prompt, [](const cail::StreamEvent& event) {
        if (const auto* delta = std::get_if<cail::TextDelta>(&event)) {
            std::cout << delta->text << std::flush;
        }
    });
    if (!response) {
        std::cerr << response.error().message << '\n';
        return 1;
    }
    std::cout << '\n';
}
