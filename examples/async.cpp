#include <cail/cail.hpp>

#include <iostream>

cail::Task<int> chat() {
  cail::Agent agent({.model = cail::openai("gpt-6-luna")});
  auto response = co_await agent.generate_async("Explain RAII in one sentence.");
  if (!response) {
    std::cerr << response.error().message << '\n';
    co_return 1;
  }
  std::cout << response->text << '\n';
  co_return 0;
}

int main() {
  return cail::run(chat());
}
