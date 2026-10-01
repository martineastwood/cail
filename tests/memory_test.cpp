#include "test_support.hpp"

#include <future>

#include <cail/agent.hpp>
#include <cail/memory.hpp>

#include <algorithm>
#include <chrono>
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

void test_trim_turns() {
  const std::vector<cail::Message> original = {
      {.role = cail::MessageRole::system, .content = {cail::TextPart{.text = "system"}}},
      {.role = cail::MessageRole::developer, .content = {cail::TextPart{.text = "developer"}}},
      {.content = {cail::TextPart{.text = "old"}}},
      {.role = cail::MessageRole::assistant, .content = {cail::TextPart{.text = "old answer"}}},
      {.content = {cail::TextPart{.text = "new"},
                   cail::ImagePart{.bytes = "image", .mime_type = "image/png"},
                   cail::PdfPart{.bytes = "pdf", .filename = "report.pdf"}}},
      {.role = cail::MessageRole::assistant,
       .tool_calls = {{.id = "one", .name = "count", .arguments = "{}"},
                      {.id = "two", .name = "count", .arguments = "{}"}}},
      {.role = cail::MessageRole::tool, .tool_call_id = "one"},
      {.role = cail::MessageRole::tool, .tool_call_id = "two"},
      {.role = cail::MessageRole::assistant,
       .tool_calls = {{.id = "three", .name = "count", .arguments = "{}"}}},
      {.role = cail::MessageRole::tool, .tool_call_id = "three"},
      {.role = cail::MessageRole::assistant, .content = {cail::TextPart{.text = "new answer"}}},
  };
  for (const auto limit : {0u, 2u, 10u}) {
    auto messages = original;
    cail::trim_turns(messages, limit);
    check(*cail::to_json(messages) == *cail::to_json(original),
          "zero or a sufficient turn limit keeps full history");
  }
  auto messages = original;
  cail::trim_turns(messages, 1);
  auto expected = original;
  expected.erase(expected.begin() + 2, expected.begin() + 4);
  check(*cail::to_json(messages) == *cail::to_json(expected),
        "trimming keeps the prefix, user attachments, and every tool round intact");
  messages.push_back({.content = {cail::TextPart{.text = "next"}}});
  messages.push_back(
      {.role = cail::MessageRole::assistant, .content = {cail::TextPart{.text = "next answer"}}});
  cail::trim_turns(messages, 1);
  check(messages.size() == 4 && messages[2].content.size() == 1 &&
            std::get<cail::TextPart>(messages[2].content.front()).text == "next",
        "trimming removes an older turn with all its attachments and tool exchanges");
  std::vector<cail::Message> empty;
  cail::trim_turns(empty, 1);
  check(empty.empty(), "empty history stays empty");
  messages.resize(2);
  cail::trim_turns(messages, 1);
  check(messages.size() == 2, "instruction-only history stays intact");
}

void test_agent_memory() {
  auto client = std::make_shared<ScriptedClient>(std::vector<cail::GenerationResponse>{
      cail::GenerationResponse{.text = "hello"},
      cail::GenerationResponse{.text = "again"},
  });
  auto memory = std::make_shared<cail::InMemoryConversationMemory>();
  cail::Agent agent({
      .model = cail::LanguageModel(
          [client](const cail::GenerationRequest& request, std::stop_token stop) {
            return client->generate(request, std::move(stop));
          }),
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
          [client](const cail::GenerationRequest& request, std::stop_token stop) {
            return client->generate(request, std::move(stop));
          }),
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
          [client](const cail::GenerationRequest& request, std::stop_token stop) {
            return client->generate(request, std::move(stop));
          }),
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
      .model =
          cail::LanguageModel([](const cail::GenerationRequest&,
                                 const std::stop_token&) -> cail::Result<cail::GenerationResponse> {
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
          [client](const cail::GenerationRequest& request, std::stop_token stop) {
            return client->generate(request, std::move(stop));
          }),
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

void test_agent_memory_keep_last_turns() {
  for (int mode = 0; mode < 4; ++mode) {
    auto client = std::make_shared<ScriptedClient>(std::vector<cail::GenerationResponse>{
        cail::GenerationResponse{.text = "one"},
        cail::GenerationResponse{.text = "two"},
        cail::GenerationResponse{.text = "three"},
    });
    auto memory = std::make_shared<cail::InMemoryConversationMemory>();
    cail::Agent agent({
        .model = cail::LanguageModel(
            [client](const cail::GenerationRequest& request, std::stop_token stop) {
              return client->generate(request, std::move(stop));
            },
            [client](const cail::GenerationRequest& request, const cail::StreamHandler& handler,
                     std::stop_token stop) { return client->stream(request, handler, stop); },
            {},
            [client](const cail::GenerationRequest& request, auto complete,
                     std::stop_token stop) -> cail::Result<void> {
              complete(client->generate(request, stop));
              return {};
            },
            [client](const cail::GenerationRequest& request, const cail::StreamHandler& handler,
                     auto complete, std::stop_token stop) -> cail::Result<void> {
              complete(client->stream(request, handler, stop));
              return {};
            }),
        .instructions = "Be concise.",
        .memory = memory,
        .conversation_id = "user-42",
    });

    check(agent.generate("first").has_value(), "the first turn succeeds");
    check(agent.generate("second").has_value(), "the second turn succeeds");
    if (mode == 0) {
      check(agent.generate("third", {.keep_last_turns = 1}).has_value(), "a bounded call succeeds");
    } else if (mode == 1) {
      check(agent.stream("third", [](const cail::StreamEvent&) {}, {.keep_last_turns = 1})
                .has_value(),
            "a bounded stream succeeds");
    } else {
      std::promise<cail::Result<cail::GenerationResponse>> completed;
      auto pending = completed.get_future();
      auto done = [&](auto result) { completed.set_value(std::move(result)); };
      auto started = mode == 2 ? agent.generate_async("third", done, {.keep_last_turns = 1})
                               : agent.stream_async("third", [](const cail::StreamEvent&) {}, done,
                                                    {.keep_last_turns = 1});
      check(started.has_value(), "a bounded async call starts");
      if (started)
        check(pending.get().has_value(), "a bounded async call succeeds");
    }
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
}

void test_agent_memory_stream() {
  auto client = std::make_shared<ScriptedClient>(std::vector<cail::GenerationResponse>{
      cail::GenerationResponse{.text = "hello"},
      cail::GenerationResponse{.text = "again"},
  });
  auto memory = std::make_shared<cail::InMemoryConversationMemory>();
  cail::Agent agent({
      .model = cail::LanguageModel(
          [client](const cail::GenerationRequest& request, std::stop_token stop) {
            return client->generate(request, std::move(stop));
          },
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

[[nodiscard]] cail::Message multimodal_message() {
  std::string binary;
  for (int byte = 0; byte < 256; ++byte) {
    binary.push_back(static_cast<char>(byte));
  }
  return cail::Message{
      .content = {cail::TextPart{.text = "Compare the image and report",
                                 .provider_options = {{"text", "kept"}}},
                  cail::ImagePart{.bytes = binary,
                                  .mime_type = "image/png",
                                  .provider_options = {{"image", "kept"}}},
                  cail::PdfPart{.bytes = "%PDF-1.7\n" + binary,
                                .filename = "report.pdf",
                                .provider_options = {{"pdf", "kept"}}}},
      .provider_options = {{"message", "kept"}},
  };
}

void check_multimodal_message(const cail::Message& actual, const cail::Message& expected) {
  const auto actual_json = cail::to_json(actual);
  const auto expected_json = cail::to_json(expected);
  check(actual_json && expected_json && *actual_json == *expected_json,
        "multimodal messages preserve binary bytes, filenames, MIME types, order, and metadata");
}

void test_multimodal_agent_memory() {
  const auto directory =
      std::filesystem::temp_directory_path() /
      ("cail_multimodal_" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  for (bool file_memory : {false, true}) {
    for (int mode = 0; mode < 3; ++mode) {
      std::filesystem::remove_all(directory);
      std::shared_ptr<cail::ConversationMemory> memory;
      if (file_memory) {
        memory = std::make_shared<cail::FileConversationMemory>(directory);
      } else {
        memory = std::make_shared<cail::InMemoryConversationMemory>();
      }
      auto client = std::make_shared<ScriptedClient>(std::vector<cail::GenerationResponse>{
          cail::GenerationResponse{
              .tool_calls = {{.id = "call-1", .name = "count", .arguments = R"({"query":"abc"})"}}},
          cail::GenerationResponse{.text = "counted"},
          cail::GenerationResponse{.text = "follow-up"},
          cail::GenerationResponse{.text = "trimmed"},
          cail::GenerationResponse{.text = "separate"},
      });
      auto model = cail::LanguageModel(
          [client](const cail::GenerationRequest& request, const std::stop_token& stop) {
            return client->generate(request, stop);
          },
          [client](const cail::GenerationRequest& request, const cail::StreamHandler& handler,
                   const std::stop_token& stop) { return client->stream(request, handler, stop); },
          {},
          [client](const cail::GenerationRequest& request,
                   const cail::LanguageModel::GenerationCompletion& complete,
                   const std::stop_token& stop) -> cail::Result<void> {
            complete(client->generate(request, stop));
            return {};
          });
      auto count = cail::tool<ToolInput, ToolOutput>("count", "Count", [](const ToolInput& input) {
        return ToolOutput{.count = static_cast<int>(input.query.size())};
      });
      cail::Agent agent({.model = model,
                         .instructions = "system",
                         .tools = {count},
                         .memory = memory,
                         .conversation_id = "conversation"});
      const auto message = multimodal_message();
      cail::Result<cail::GenerationResponse> first;
      std::string deltas;
      if (mode == 0) {
        first = agent.generate(message);
      } else if (mode == 1) {
        first = agent.stream(message, [&](const cail::StreamEvent& event) {
          if (const auto* text = std::get_if<cail::TextDelta>(&event)) {
            deltas += text->text;
          }
        });
      } else {
        std::promise<cail::Result<cail::GenerationResponse>> completed;
        auto pending = completed.get_future();
        check(agent
                  .generate_async(message,
                                  [&](auto result) { completed.set_value(std::move(result)); })
                  .has_value(),
              "multimodal async agent starts");
        first = pending.get();
      }
      check(first && first->text == "counted" && first->tool_results.size() == 1,
            "multimodal agent inputs run the tool loop in all execution modes");
      check(mode != 1 || deltas == "counted", "multimodal streaming forwards output events");
      check(client->requests.size() == 2, "multimodal request makes a tool follow-up");
      if (client->requests.size() != 2) {
        continue;
      }
      check_multimodal_message(client->requests[0].messages[1], message);
      check_multimodal_message(client->requests[1].messages[1], message);
      if (file_memory) {
        memory = std::make_shared<cail::FileConversationMemory>(directory);
        std::ifstream file(directory / "conversation.json", std::ios::binary);
        const std::string json{std::istreambuf_iterator<char>{file},
                               std::istreambuf_iterator<char>{}};
        check(json.find(cail::detail::base64_encode(
                  std::get<cail::ImagePart>(message.content[1]).bytes)) != std::string::npos,
              "file memory encodes binary attachments as base64");
      }
      auto stored = memory->load("conversation");
      check(stored && stored->size() == 4,
            "memory stores the attachment input once and every tool round");
      if (!stored || stored->empty()) {
        continue;
      }
      check_multimodal_message(stored->front(), message);
      cail::Agent resumed({.model = model,
                           .instructions = "system",
                           .tools = {count},
                           .memory = memory,
                           .conversation_id = "conversation"});
      check(resumed.generate("What about the report?").has_value(),
            "a text follow-up recalls attachments");
      check(client->requests.back().messages.size() == 6,
            "follow-up replays the full multimodal turn");
      check_multimodal_message(client->requests.back().messages[1], message);
      check(resumed.generate("Continue", {.keep_last_turns = 1}).has_value(),
            "multimodal history can be trimmed");
      const auto& trimmed = client->requests.back().messages;
      check(trimmed.size() == 4 &&
                std::ranges::all_of(trimmed,
                                    [](const cail::Message& item) {
                                      return std::ranges::all_of(
                                          item.content, [](const cail::ContentPart& part) {
                                            return std::holds_alternative<cail::TextPart>(part);
                                          });
                                    }),
            "trimming drops attachments with their original user message");
      check(memory->load("conversation")->size() == 8,
            "trimming preserves complete stored multimodal history");
      check(resumed.generate(message, {.conversation_id = "other"}).has_value(),
            "multimodal inputs support conversation overrides");
      check(client->requests.back().messages.size() == 2 && memory->load("other")->size() == 2,
            "attachments remain isolated by conversation id");
      check(memory->clear("conversation").has_value() && memory->load("conversation")->empty(),
            "clearing removes multimodal history");
    }
  }
  std::filesystem::remove_all(directory);
}

void test_multimodal_input_validation() {
  int calls = 0;
  auto model = cail::LanguageModel(
      [&](const cail::GenerationRequest&,
          const std::stop_token&) -> cail::Result<cail::GenerationResponse> {
        ++calls;
        return std::unexpected(cail::Error{.code = cail::ErrorCode::provider_response});
      },
      [&](const cail::GenerationRequest&, const cail::StreamHandler&,
          const std::stop_token&) -> cail::Result<cail::GenerationResponse> {
        ++calls;
        return std::unexpected(cail::Error{.code = cail::ErrorCode::provider_response});
      },
      {},
      [&](const cail::GenerationRequest&, const cail::LanguageModel::GenerationCompletion& complete,
          const std::stop_token&) -> cail::Result<void> {
        ++calls;
        complete(std::unexpected(cail::Error{.code = cail::ErrorCode::provider_response}));
        return {};
      });
  auto memory = std::make_shared<cail::InMemoryConversationMemory>();
  cail::Agent agent({.model = model, .memory = memory, .conversation_id = "conversation"});
  const auto ignore = [](const cail::StreamEvent&) {};
  for (const auto role : {cail::MessageRole::system, cail::MessageRole::developer,
                          cail::MessageRole::assistant, cail::MessageRole::tool}) {
    auto message = multimodal_message();
    message.role = role;
    int callbacks = 0;
    check(!agent.generate(message) && !agent.stream(message, ignore) &&
              !agent.generate_async(message, [&](auto) { ++callbacks; }) && callbacks == 0,
          "memory input rejects non-user roles in all execution modes");
  }
  auto message = multimodal_message();
  message.tool_call_id = "call";
  check(!agent.generate(message), "a user message cannot contain a tool result ID");
  message.tool_call_id.clear();
  message.tool_calls = {{.id = "call", .name = "tool", .arguments = "{}"}};
  check(!agent.generate(message), "a user message cannot contain tool calls");
  check(!agent.generate(cail::Message{}), "empty user messages are rejected");
  check(calls == 0, "invalid memory input never reaches the model");
  check(!agent.generate(multimodal_message()) && !agent.stream(multimodal_message(), ignore),
        "multimodal model failures propagate");
  std::promise<cail::Result<cail::GenerationResponse>> completed;
  auto completion = completed.get_future();
  check(agent
            .generate_async(multimodal_message(),
                            [&](auto result) { completed.set_value(std::move(result)); })
            .has_value(),
        "an async multimodal request starts before memory loading finishes");
  check(!completion.get(), "a started async multimodal request reports failure through completion");
  check(memory->load("conversation")->empty(), "failed multimodal calls store no attachments");
  std::stop_source stop;
  stop.request_stop();
  const auto cancelled = agent.generate(multimodal_message(), {.stop = stop.get_token()});
  check(!cancelled && cancelled.error().code == cail::ErrorCode::cancelled && calls == 3,
        "cancelled multimodal input stores nothing and does not reach the model");
}

void test_attachment_base64() {
  for (const std::string& bytes : {std::string{}, std::string{"a"}, std::string{"ab"},
                                   std::string{"abc"}, std::string{"\0\xff", 2}}) {
    const auto encoded = cail::detail::base64_encode(bytes);
    const auto decoded = cail::detail::base64_decode(encoded);
    check(decoded && *decoded == bytes, "base64 decoding preserves binary and padded payloads");
  }
  for (const auto invalid : {"a", "!!!!", "=AAA", "AA=A", "AA==AAAA", "AB==", "AAB="}) {
    check(!cail::detail::base64_decode(invalid),
          "invalid or noncanonical attachment base64 is rejected");
  }
  const auto directory =
      std::filesystem::temp_directory_path() /
      ("cail_bad_attachment_" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directory(directory);
  {
    std::ofstream file(directory / "bad.json");
    file
        << R"({"version":2,"messages":[{"role":"user","content":[{"bytes":"!!!!","mime_type":"image/png","provider_options":{}}],"tool_call_id":"","tool_calls":[],"provider_options":{}}]})";
  }
  cail::FileConversationMemory memory(directory);
  const auto invalid = memory.load("bad");
  check(!invalid && invalid.error().code == cail::ErrorCode::memory,
        "corrupted encoded attachments return a memory error");
  {
    std::ofstream file(directory / "old.json");
    file << R"({"version":1,"messages":[]})";
  }
  check(!memory.load("old"), "old conversation formats are rejected");
  std::filesystem::remove_all(directory);
}

void test_paused_turn_not_saved() {
  auto memory = std::make_shared<cail::InMemoryConversationMemory>();
  cail::LanguageModel model([](auto, auto) -> cail::Result<cail::GenerationResponse> {
    return cail::GenerationResponse{
        .tool_calls = {{.id = "a", .name = "lookup", .arguments = "1"}}};
  });
  auto tool = cail::tool<int, int>("lookup", "Lookup", [](int input) { return input; });
  cail::Agent agent(
      {.model = model, .tools = {tool}, .memory = memory, .conversation_id = "paused"});
  auto paused = agent.generate("look up", {.pause_when = [](const auto&) { return true; }});
  check(paused && paused->tool_continuation && memory->load("paused")->empty(),
        "paused turns do not save incomplete tool exchanges");
  check(agent.generate("try again", {.pause_when = [](const auto&) { return true; }}).has_value(),
        "paused turns release the conversation lease");
}

} // namespace test

int main() {
  test::test_paused_turn_not_saved();
  test::test_in_memory_conversation();
  test::test_file_conversation();
  test::test_trim_turns();
  test::test_agent_memory();
  test::test_agent_memory_conversation_ids();
  test::test_agent_memory_tool_rounds();
  test::test_agent_memory_failures_store_nothing();
  test::test_agent_memory_explicit_requests_bypass_memory();
  test::test_agent_memory_keep_last_turns();
  test::test_agent_memory_stream();
  test::test_multimodal_agent_memory();
  test::test_multimodal_input_validation();
  test::test_attachment_base64();
  return test::failures == 0 ? 0 : 1;
}
