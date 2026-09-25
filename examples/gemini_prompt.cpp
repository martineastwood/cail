#include <cail/cail.hpp>

#include <iostream>

int main()
{
    const auto result = cail::generate_text({
        .model = cail::gemini("gemini-3.5-flash-lite"),
        .prompt = "Reply with exactly OK.",
    });
    if (!result) {
        std::cerr << result.error().message << '\n';
        return 1;
    }
    std::cout << result->text << '\n';
}
