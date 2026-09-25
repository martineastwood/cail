#include <cail/cail.hpp>

#include <cstdlib>
#include <iostream>

int main()
{
    const char* key = std::getenv("OPENROUTER_API_KEY");
    if (!key || !*key) {
        std::cerr << "Set OPENROUTER_API_KEY before running this example.\n";
        return 1;
    }

    const auto response = cail::openrouter("openai/gpt-4o-mini").generate(cail::GenerationRequest{
        .messages = {cail::Message{
            .role = cail::MessageRole::user,
            .content = {cail::TextPart{.text = "Name one advantage of native C++ AI applications in one sentence."}},
        }},
    });
    if (!response) {
        std::cerr << response.error().message << '\n';
        return 1;
    }
    std::cout << response->text << '\n';
}
