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

    const auto response = cail::generate_text({
        .model = cail::create_anthropic({.max_tokens = 128})("claude-haiku-4-5-20251001"),
        .prompt = "Name one advantage of native C++ AI applications in one sentence.",
    });
    if (!response) {
        std::cerr << response.error().message << '\n';
        return 1;
    }
    std::cout << response->text << '\n';
}
