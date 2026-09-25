#include <cail/cail.hpp>

#include <cstdlib>
#include <iostream>
#include <string>

struct Answer {
    std::string language;
};

int main()
{
    const char* key = std::getenv("ANTHROPIC_API_KEY");
    if (!key || !*key) {
        std::cerr << "Set ANTHROPIC_API_KEY before running this example.\n";
        return 1;
    }

    const auto answer = cail::generate_object<Answer>({
        .model = cail::anthropic("claude-haiku-4-5-20251001"),
        .prompt = "What programming language is CAIL, a C++ AI SDK, written in?",
    });
    if (!answer) {
        std::cerr << answer.error().message << '\n';
        return 1;
    }
    std::cout << answer->language << '\n';
}
