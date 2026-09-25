#include <cail/cail.hpp>

#include <cstdlib>
#include <iostream>

int main()
{
    const char* key = std::getenv("ANTHROPIC_API_KEY");
    if (!key || !*key) {
        std::cerr << "Set ANTHROPIC_API_KEY before running this example.\n";
        return 1;
    }

    const auto response = cail::anthropic("claude-haiku-4-5-20251001").stream(
        "Name one advantage of native C++ AI applications in one sentence.",
        [](const cail::StreamEvent& event) {
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
