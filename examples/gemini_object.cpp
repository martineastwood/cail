#include <cail/cail.hpp>

#include <cstdlib>
#include <iostream>
#include <string>

struct Answer {
    std::string language;
};

int main()
{
    const char* key = std::getenv("GEMINI_API_KEY");
    if (!key || !*key) {
        std::cerr << "Set GEMINI_API_KEY before running this example.\n";
        return 1;
    }

    const auto answer = cail::generate_object<Answer>({
        .model = cail::gemini("gemini-3.5-flash-lite"),
        .prompt = "Return the language CAIL is written in. It is C++.",
    });
    if (!answer) {
        std::cerr << answer.error().message << '\n';
        return 1;
    }
    std::cout << answer->language << '\n';
}
