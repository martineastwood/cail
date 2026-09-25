#include <cail/cail.hpp>

#include <iostream>
#include <string>

struct Answer {
    std::string language;
};

int main()
{
    const auto answer = cail::generate_object<Answer>({
        .model = cail::local("qwen3-1.7b"),
        .prompt = "What programming language is CAIL, a C++ AI SDK, written in?",
    });
    if (!answer) {
        std::cerr << answer.error().message << '\n';
        return 1;
    }
    std::cout << answer->language << '\n';
}