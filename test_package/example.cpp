#include <cail/cail.hpp>

#include <iostream>
#include <string>
#include <vector>

enum class Sentiment {
  positive,
  neutral,
  negative,
};

struct Analysis {
  cail::Field<Sentiment> sentiment{
      .description = "Overall sentiment",
  };
  cail::Field<double> confidence{
      .description = "Confidence score",
      .minimum = 0.0,
      .maximum = 1.0,
  };
  std::vector<std::string> topics;
};

int main() {
  const auto schema = cail::json_schema<Analysis>();
  if (!schema) {
    std::cerr << schema.error().message << '\n';
    return 1;
  }
  const auto model = cail::openai("test-model");
  if (!model || !model.adapter_capabilities().structured_output) {
    return 1;
  }
  if (schema->find("confidence") == std::string::npos ||
      schema->find("negative") == std::string::npos) {
    std::cerr << "generated schema is missing expected fields: " << *schema << '\n';
    return 1;
  }
  std::cout << *schema << '\n';
  return 0;
}
