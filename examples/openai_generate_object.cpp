#include <cail/cail.hpp>

#include <iostream>

enum class Sentiment {
    Positive,
    Neutral,
    Negative,
};

struct Analysis {
    cail::Field<Sentiment> sentiment{.description = "Overall sentiment"};

    cail::Field<double> confidence{
        .description = "Confidence score",
        .minimum = 0.0,
        .maximum = 1.0,
    };
};

int main() {
    const auto analysis = cail::generate_object<Analysis>({
        .model = cail::openai("gpt-6-luna"),
        .prompt = "Classify the sentiment of this review and give a confidence score: "
                  "The package arrived quickly and the support team was helpful.",
    });
    if (!analysis) {
        std::cerr << analysis.error().message << '\n';
        return 1;
    }
    const auto structured_output = cail::to_json(*analysis);
    if (!structured_output) {
        std::cerr << structured_output.error().message << '\n';
        return 1;
    }
    std::cout << *structured_output << '\n';
}
