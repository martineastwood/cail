#include <cail/cail.hpp>

#include <iostream>

int main()
{
    const auto result = cail::generate_text({
        .model = cail::ollama_cloud("gemma4:31b"),
        .prompt = "Reply with exactly OK.",
    });
    if (!result) {
        std::cerr << result.error().message << '\n';
        return 1;
    }
    std::cout << result->text << '\n';
}
