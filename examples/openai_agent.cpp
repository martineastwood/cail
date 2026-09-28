#include <cail/agent.hpp>
#include <cail/openai.hpp>

#include <iostream>

int main() {
  cail::Agent agent({
      .model = cail::openai("gpt-6-luna"),
      .instructions = "You are a concise research assistant.",
  });
  auto response = agent.generate("Name one benefit of native C++ AI applications.");
  if (!response) {
    std::cerr << response.error().message << '\n';
    return 1;
  }
  std::cout << response->text << '\n';
}
