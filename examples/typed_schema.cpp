#include <cail/cail.hpp>

#include <iostream>
#include <string>
#include <vector>

enum class Sentiment {
    Positive,
    Neutral,
    Negative,
};

struct Analysis {
    cail::Field<Sentiment> sentiment{
        .description = "Overall sentiment"
    };

    cail::Field<double> confidence{
        .description = "Confidence score",
        .minimum = 0.0,
        .maximum = 1.0,
    };

    std::vector<std::string> topics;
};

int main()
{
    Analysis analysis{
        .sentiment = {.value = Sentiment::Positive},
        .confidence = {.value = 0.94},
        .topics = {"shipping", "support"},
    };

    const auto encoded = cail::to_json(analysis);
    if (!encoded) {
        std::cerr << encoded.error().message << '\n';
        return 1;
    }

    const auto decoded = cail::from_json<Analysis>(*encoded);
    if (!decoded) {
        std::cerr << decoded.error().message << '\n';
        return 1;
    }

    const auto output_schema = cail::json_schema<Analysis>();
    if (!output_schema) {
        std::cerr << output_schema.error().message << '\n';
        return 1;
    }

    std::cout << *encoded << '\n';
    std::cout << *output_schema << '\n';
}
