#include <cail/cail.hpp>

#include <iostream>

int main()
{
    auto opencode = cail::create_opencode({
        .service = cail::OpenCodeService::zen,
    });
    auto response = cail::generate_text({
        .model = opencode("gpt-6-luna", cail::OpenCodeApiFamily::responses),
        .prompt = "Explain one benefit of native C++ applications in one sentence.",
        .session_id = "example-conversation-1",
    });
    if (!response) {
        std::cerr << response.error().message << '\n';
        return 1;
    }
    std::cout << response->text << '\n';
}
