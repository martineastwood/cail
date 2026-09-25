#include <cail/cail.hpp>

#include <iostream>

int main()
{
    const auto result = cail::generate_text({
        .model = cail::local("qwen3-1.7b"),
        .prompt = "Reply with exactly OK.",
    });
    if (!result) {
        std::cerr << result.error().message << '\n';
        return 1;
    }
    std::cout << result->text << '\n';
}