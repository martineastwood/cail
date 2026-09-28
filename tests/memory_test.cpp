#include "test_support.hpp"

#include <cail/agent.hpp>
#include <cail/memory.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stop_token>
#include <utility>
#include <variant>

namespace test {

void test_in_memory_conversation() {
  cail::InMemoryConversationMemory memory;
  auto empty = memory.load("user-42");
  check(empty && empty->empty(), "an unknown conversation loads empty");

  check(memory.append("user-42", {cail::Message{.content = {cail::TextPart{.text = "hi"}}}})
            .has_value(),
        "append succeeds");
  auto loaded = memory.load("user-42");
  check(loaded && loaded->size() == 1 &&
            std::get<cail::TextPart>(loaded->front().content.front()).text == "hi",
        "append stores the message");
  check(memory.append("user-42", {cail::Message{.content = {cail::TextPart{.text = "again"}}}})
            .has_value(),
        "a second append succeeds");
  loaded = memory.load("user-42");
  check(loaded && loaded->size() == 2, "appends accumulate in order");
  check(memory.clear("user-42").has_value(), "clear succeeds");
  check(memory.load("user-42") && memory.load("user-42")->empty(),
        "clear empties the conversation");
}

void test_file_conversation() {
  const auto directory = std::filesystem::temp_directory_path() / "cail_memory_test";
  std::filesystem::remove_all(directory);
  cail::FileConversationMemory memory(directory);

  auto missing = memory.load("user-42");
  check(missing && missing->empty(), "a conversation with no file loads empty");

  check(memory.append("user-42", {cail::Message{.content = {cail::TextPart{.text = "hi"}}}})
            .has_value(),
        "append writes the conversation file");
  check(std::filesystem::exists(directory / "user-42.json"), "one file per conversation id");

  auto loaded = memory.load("user-42");
  check(loaded && loaded->size() == 1 &&
            std::get<cail::TextPart>(loaded->front().content.front()).text == "hi",
        "a conversation survives a restart from disk");

  check(memory.append("odd/id:42", {cail::Message{.content = {cail::TextPart{.text = "safe"}}}})
            .has_value(),
        "unsafe conversation ids still append");
  check(std::filesystem::exists(directory / "odd_id_42.json"),
        "conversation ids sanitize into safe file names");

  {
    std::ofstream broken(directory / "broken.json", std::ios::trunc);
    broken << "not json";
  }
  check(!memory.load("broken").has_value(), "a corrupt file reports an error");

  {
    std::ofstream future(directory / "future.json", std::ios::trunc);
    future << R"({"version":9,"messages":[]})";
  }
  auto future = memory.load("future");
  check(!future && future.error().code == cail::ErrorCode::memory,
        "an unsupported file version reports a memory error");

  check(memory.clear("user-42").has_value() && !std::filesystem::exists(directory / "user-42.json"),
        "clear removes the conversation file");
  std::filesystem::remove_all(directory);
}

void test_trim_messages() {
  cail::ToolCall call{.id = "call-1", .name = "count", .arguments = "{}"};
  std::vector<cail::Message> messages = {
      cail::Message{.role = cail::MessageRole::system, .content = {cail::TextPart{.text = "S"}}},
      cail::Message{.content = {cail::TextPart{.text = "u1"}}},
      cail::Message{.role = cail::MessageRole::assistant, .tool_calls = {call}},
      cail::Message{.role = cail::MessageRole::tool, .tool_call_id = "call-1"},
      cail::Message{.content = {cail::TextPart{.text = "u2"}}},
      cail::Message{.role = cail::MessageRole::assistant,
                    .content = {cail::TextPart{.text = "a2"}}},
  };

  cail::trim_messages(messages, 4);
  check(messages.size() == 5 && messages.front().role == cail::MessageRole::system &&
            messages.back().role == cail::MessageRole::assistant,
        "trimming keeps the system prefix plus the requested trailing messages");

  messages = {
      cail::Message{.role = cail::MessageRole::system, .content = {cail::TextPart{.text = "S"}}},
      cail::Message{.content = {cail::TextPart{.text = "u1"}}},
      cail::Message{.role = cail::MessageRole::assistant, .tool_calls = {call}},
      cail::Message{.role = cail::MessageRole::tool, .tool_call_id = "call-1"},
      cail::Message{.content = {cail::TextPart{.text = "u2"}}},
      cail::Message{.role = cail::MessageRole::assistant,
                    .content = {cail::TextPart{.text = "a2"}}},
  };
  cail::trim_messages(messages, 3);
  check(messages.size() == 3, "trimming drops evicted messages");
  check(messages[0].role == cail::MessageRole::system &&
            std::get<cail::TextPart>(messages[0].content.front()).text == "S",
        "the leading system message survives trimming");
  check(messages[1].role == cail::MessageRole::user &&
            std::get<cail::TextPart>(messages[1].content.front()).text == "u2",
        "trimming keeps the newest messages");
  check(std::none_of(
            messages.begin(), messages.end(),
            [](const cail::Message& message) { return message.role == cail::MessageRole::tool; }),
        "trimming drops tool results whose tool call was evicted");

  cail::trim_messages(messages, 0);
  check(messages.size() == 1 && messages.front().role == cail::MessageRole::system,
        "a zero window keeps only the system prefix");
}

void test_agent_memory() {
  auto client = std::make_shared<ScriptedClient>(std::vector<cail::GenerationResponse>{
      cail::GenerationResponse{.text = "hello"},
      cail::GenerationResponse{.text = "again"},
  });
  auto memory = std::make_shared<cail::InMemoryConversationMemory>();
  cail::Agent agent({
      .model = cail::LanguageModel(
          [client](const cail::GenerationRequest& request) { return client->generate(request); }),
      .instructions = "Be concise.",
      .memory = memory,
      .conversation_id = "user-42",
  });

  auto first = agent.generate("Say hello.");
  check(first && first->text == "hello", "a memory agent returns the model response");
  check(client->requests.size() == 1 && client->requests.front().messages.size() == 2,
        "the first call sends only the instructions and the prompt");

  auto second = agent.generate("What did I say?");
  check(second && second->text == "again", "the second call succeeds");
  const auto& sent = client->requests[1].messages;
  check(sent.size() == 4, "the second call replays the stored conversation");
  check(sent[0].role == cail::MessageRole::system &&
            std::get<cail::TextPart>(sent[0].content.front()).text == "Be concise.",
        "instructions stay the leading message");
  check(sent[1].role == cail::MessageRole::user &&
            std::get<cail::TextPart>(sent[1].content.front()).text == "Say hello.",
        "the stored conversation contains the first prompt");
  check(sent[2].role == cail::MessageRole::assistant &&
            std::get<cail::TextPart>(sent[2].content.front()).text == "hello",
        "the stored conversation contains the first reply");
  check(sent[3].role == cail::MessageRole::user &&
            std::get<cail::TextPart>(sent[3].content.front()).text == "What did I say?",
        "the second prompt follows the stored conversation");

  auto stored = memory->load("user-42");
  check(stored && stored->size() == 4, "memory stores the prompt and the turn for each call");
  check(first->turn.size() == 1 &&
            std::get<cail::TextPart>(first->turn.front().content.front()).text == "hello",
        "the response reports the messages the turn added");
}

void test_agent_memory_conversation_ids() {
  auto client = std::make_shared<ScriptedClient>(std::vector<cail::GenerationResponse>{
      cail::GenerationResponse{.text = "a"},
      cail::GenerationResponse{.text = "b"},
      cail::GenerationResponse{.text = "c"},
  });
  auto memory = std::make_shared<cail::InMemoryConversationMemory>();
  cail::Agent agent({
      .model = cail::LanguageModel(
          [client](const cail::GenerationRequest& request) { return client->generate(request); }),
      .memory = memory,
  });

  check(agent.generate("one", {.conversation_id = "first"}).has_value(),
        "a per-call conversation id stores under that id");
  check(agent.generate("two", {.conversation_id = "second"}).has_value(),
        "a different conversation id starts a fresh conversation");
  auto first = memory->load("first");
  check(first && first->size() == 2, "each conversation id keeps its own history");
  auto second = memory->load("second");
  check(second && second->size() == 2, "conversations stay isolated");
  check(agent.generate("three").has_value(), "a call without a conversation id still succeeds");
  check(memory->load("first") && memory->load("first")->size() == 2,
        "a call without a conversation id stores nothing");
}

void test_agent_memory_tool_rounds() {
  auto client = std::make_shared<ScriptedClient>(std::vector<cail::GenerationResponse>{
      cail::GenerationResponse{
          .tool_calls = {cail::ToolCall{
              .id = "call-1", .name = "count", .arguments = R"({"query":"abc"})"}},
      },
      cail::GenerationResponse{.text = "counted"},
  });
  auto memory = std::make_shared<cail::InMemoryConversationMemory>();
  cail::Agent agent({
      .model = cail::LanguageModel(
          [client](const cail::GenerationRequest& request) { return client->generate(request); }),
      .tools = {cail::tool<ToolInput, ToolOutput>("count", "Count characters",
                                                  [](const ToolInput& input) {
                                                    return ToolOutput{.count = static_cast<int>(
                                                                          input.query.size())};
                                                  })},
      .memory = memory,
      .conversation_id = "user-42",
  });

  auto response = agent.generate("count abc");
  check(response && response->text == "counted", "a tool-loop turn succeeds");
  auto stored = memory->load("user-42");
  check(stored && stored->size() == 4, "memory stores every tool round of the turn");
  check(stored && stored->at(1).role == cail::MessageRole::assistant &&
            stored->at(1).tool_calls.size() == 1 && stored->at(1).tool_calls.front().id == "call-1",
        "the stored turn contains the assistant tool-call message");
  check(stored && stored->at(2).role == cail::MessageRole::tool &&
            stored->at(2).tool_call_id == "call-1" &&
            std::get<cail::TextPart>(stored->at(2).content.front()).text == R"({"count":3})",
        "the stored turn contains the tool result");
  check(stored && stored->at(3).role == cail::MessageRole::assistant &&
            std::get<cail::TextPart>(stored->at(3).content.front()).text == "counted",
        "the stored turn contains the final reply");
  check(client->requests.size() == 2 && client->requests[1].messages.size() == 3,
        "the follow-up call replays the stored conversation plus the new prompt");
}

void test_agent_memory_failures_store_nothing() {
  auto memory = std::make_shared<cail::InMemoryConversationMemory>();
  cail::Agent agent({
      .model = cail::LanguageModel(
          [](const cail::GenerationRequest&) -> cail::Result<cail::GenerationResponse> {
            return std::unexpected(
                cail::Error{.code = cail::ErrorCode::provider_response, .message = "down."});
          }),
      .memory = memory,
      .conversation_id = "user-42",
  });

  auto response = agent.generate("Say hello.");
  check(!response.has_value(), "a failed call reports the model error");
  auto stored = memory->load("user-42");
  check(stored && stored->empty(), "a failed call stores nothing");
}

void test_agent_memory_explicit_requests_bypass_memory() {
  auto client = std::make_shared<ScriptedClient>(
      std::vector<cail::GenerationResponse>{cail::GenerationResponse{.text = "hello"}});
  auto memory = std::make_shared<cail::InMemoryConversationMemory>();
  cail::Agent agent({
      .model = cail::LanguageModel(
          [client](const cail::GenerationRequest& request) { return client->generate(request); }),
      .instructions = "Be concise.",
      .memory = memory,
      .conversation_id = "user-42",
  });

  auto response = agent.generate(cail::GenerationRequest{
      .messages = {cail::Message{.content = {cail::TextPart{.text = "Say hello."}}}}});
  check(response && response->text == "hello", "an explicit request succeeds");
  check(client->requests.front().messages.size() == 2,
        "an explicit request sends exactly the caller history plus instructions");
  auto stored = memory->load("user-42");
  check(stored && stored->empty(), "an explicit request loads and stores nothing");
}

void test_agent_memory_keep_last_messages() {
  auto client = std::make_shared<ScriptedClient>(std::vector<cail::GenerationResponse>{
      cail::GenerationResponse{.text = "one"},
      cail::GenerationResponse{.text = "two"},
      cail::GenerationResponse{.text = "three"},
  });
  auto memory = std::make_shared<cail::InMemoryConversationMemory>();
  cail::Agent agent({
      .model = cail::LanguageModel(
          [client](const cail::GenerationRequest& request) { return client->generate(request); }),
      .instructions = "Be concise.",
      .memory = memory,
      .conversation_id = "user-42",
  });

  check(agent.generate("first").has_value(), "the first turn succeeds");
  check(agent.generate("second").has_value(), "the second turn succeeds");
  check(agent.generate("third", {.keep_last_messages = 2}).has_value(), "a bounded call succeeds");
  const auto& sent = client->requests[2].messages;
  check(sent.size() == 4, "a bounded call sends the instructions plus the window");
  check(sent[1].role == cail::MessageRole::user &&
            std::get<cail::TextPart>(sent[1].content.front()).text == "second",
        "the window keeps the oldest retained messages");
  check(sent[2].role == cail::MessageRole::assistant &&
            std::get<cail::TextPart>(sent[2].content.front()).text == "two",
        "the window keeps complete turns");
  auto stored = memory->load("user-42");
  check(stored && stored->size() == 6, "trimming shapes what is sent, not what is stored");
}

void test_agent_memory_stream() {
  auto client = std::make_shared<ScriptedClient>(std::vector<cail::GenerationResponse>{
      cail::GenerationResponse{.text = "hello"},
      cail::GenerationResponse{.text = "again"},
  });
  auto memory = std::make_shared<cail::InMemoryConversationMemory>();
  cail::Agent agent({
      .model = cail::LanguageModel(
          [client](const cail::GenerationRequest& request) { return client->generate(request); },
          [client](const cail::GenerationRequest& request, const cail::StreamHandler& on_event,
                   std::stop_token stop) {
            return client->stream(request, on_event, std::move(stop));
          }),
      .memory = memory,
      .conversation_id = "user-42",
  });

  std::string streamed;
  auto first = agent.stream("Say hello.", [&](const cail::StreamEvent& event) {
    if (const auto* delta = std::get_if<cail::TextDelta>(&event)) {
      streamed += delta->text;
    }
  });
  check(first && first->text == "hello" && streamed == "hello", "a memory agent streams");
  auto second = agent.generate("What did I say?");
  check(second && second->text == "again" && client->requests[1].messages.size() == 3,
        "streamed turns replay in the next call");
}

} // namespace test

int main() {
  test::test_in_memory_conversation();
  test::test_file_conversation();
  test::test_trim_messages();
  test::test_agent_memory();
  test::test_agent_memory_conversation_ids();
  test::test_agent_memory_tool_rounds();
  test::test_agent_memory_failures_store_nothing();
  test::test_agent_memory_explicit_requests_bypass_memory();
  test::test_agent_memory_keep_last_messages();
  test::test_agent_memory_stream();
  return test::failures == 0 ? 0 : 1;
}
