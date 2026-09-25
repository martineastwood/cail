#include <cail/cail.hpp>

#include <iostream>
#include <string>

struct WordInput {
    std::string word;
};

struct WordOutput {
    int length{};
};

int main() {
    auto word_length = cail::tool<WordInput, WordOutput>(
        "word_length", "Count the letters in a word.",
        [](const WordInput& input) { return WordOutput{.length = static_cast<int>(input.word.size())}; });

    const auto result = cail::generate_text({
        .model = cail::gemini("gemini-3.5-flash-lite"),
        .prompt = "Use the word_length tool to count the letters in 'native', then answer with the count.",
        .tools = {word_length},
    });
    if (!result) {
        std::cerr << result.error().message << '\n';
        return 1;
    }
    if (result->tool_results.empty()) {
        std::cerr << "Gemini did not call word_length.\n";
        return 1;
    }
    const auto count = word_length.decode_output(result->tool_results.front());
    if (!count) {
        std::cerr << count.error().message << '\n';
        return 1;
    }
    std::cout << "Tool result: " << count->length << '\n';
    std::cout << "Answer: " << result->text << '\n';
}
