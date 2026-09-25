#include <cail/cail.hpp>

#include <cstdlib>
#include <iostream>

int main() {
    const auto* api_key = std::getenv("OPENAI_API_KEY");
    if (!api_key) {
        std::cerr << "Set OPENAI_API_KEY before running this example.\n";
        return 1;
    }

    auto openai = cail::create_openai({
        .api_key = api_key,
    });

    const auto text = cail::generate_text({
        .model = openai("gpt-6-luna"),
        .system = "You are a concise and practical programming assistant.",
        .prompt = "Give me one practical tip for writing clearer code.",
    });
    if (!text) {
        std::cerr << text.error().message << '\n';
        return 1;
    }
    if (text->status != cail::GenerationStatus::completed) {
        std::cerr << "OpenAI did not complete the text response: " << text->text << '\n';
        return 1;
    }
    std::cout << text->text << '\n';
}
