#include <cail/agent.hpp>
#include <cail/memory.hpp>
#include <cail/openai.hpp>

#include <iostream>
#include <memory>

int main() {
  cail::Agent agent({
      .model = cail::openai("gpt-6-luna"),
      .instructions = "You are a concise research assistant.",
      .memory = std::make_shared<cail::FileConversationMemory>("conversations"),
      .conversation_id = "user-42",
  });

  auto response = agent.generate("My favorite language is C++. Remember that.");
  if (!response) {
    std::cerr << response.error().message << '\n';
    return 1;
  }
  std::cout << response->text << "\n\n";

  // The second call replays the stored conversation, so the model remembers.
  response = agent.generate("What is my favorite language?");
  if (!response) {
    std::cerr << response.error().message << '\n';
    return 1;
  }
  std::cout << response->text << '\n';
}
