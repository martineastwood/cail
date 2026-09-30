#pragma once

#include <cail/detail/base64.hpp>
#include <cail/error.hpp>
#include <cail/generation.hpp>
#include <cail/json.hpp>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cail {

// Bounds a message list to the last keep_last messages, preserving leading
// system and developer messages and dropping tool results whose paired tool
// call was trimmed away.
inline void trim_messages(std::vector<Message>& messages, std::size_t keep_last) {
  std::size_t prefix = 0;
  while (prefix < messages.size() && (messages[prefix].role == MessageRole::system ||
                                      messages[prefix].role == MessageRole::developer)) {
    ++prefix;
  }
  if (messages.size() <= prefix + keep_last) {
    return;
  }
  messages.erase(messages.begin() + static_cast<std::ptrdiff_t>(prefix),
                 messages.end() - static_cast<std::ptrdiff_t>(keep_last));
  std::unordered_set<std::string> retained_calls;
  for (const auto& message : messages) {
    for (const auto& call : message.tool_calls) {
      retained_calls.insert(call.id);
    }
  }
  std::erase_if(messages, [&retained_calls](const Message& message) {
    return message.role == MessageRole::tool && !retained_calls.contains(message.tool_call_id);
  });
}

namespace detail {

// File names cannot contain path separators or control characters; keep a
// conservative ASCII subset and map everything else to '_'.
inline std::string sanitize_conversation_id(const std::string& conversation_id) {
  std::string name;
  name.reserve(conversation_id.size());
  for (const char character : conversation_id) {
    const bool safe = (character >= 'a' && character <= 'z') ||
                      (character >= 'A' && character <= 'Z') ||
                      (character >= '0' && character <= '9') || character == '_' ||
                      character == '-' || character == '.';
    name.push_back(safe ? character : '_');
  }
  return name.empty() ? std::string{"_"} : name;
}

} // namespace detail

// Storage for agent conversation history. Implement this over your own store
// for durable sessions.
class ConversationMemory {
public:
  virtual ~ConversationMemory() = default;

  [[nodiscard]] virtual Result<std::vector<Message>> load(const std::string& conversation_id) = 0;

  [[nodiscard]] virtual Result<void> append(const std::string& conversation_id,
                                            std::vector<Message> messages) = 0;

  [[nodiscard]] virtual Result<void> clear(const std::string& conversation_id) = 0;
};

// Conversation history in process memory. Forgets everything on restart.
class InMemoryConversationMemory final : public ConversationMemory {
public:
  [[nodiscard]] Result<std::vector<Message>> load(const std::string& conversation_id) override {
    if (const auto found = conversations_.find(conversation_id); found != conversations_.end()) {
      return found->second;
    }
    return std::vector<Message>{};
  }

  Result<void> append(const std::string& conversation_id, std::vector<Message> messages) override {
    auto& conversation = conversations_[conversation_id];
    conversation.insert(conversation.end(), std::make_move_iterator(messages.begin()),
                        std::make_move_iterator(messages.end()));
    return {};
  }

  Result<void> clear(const std::string& conversation_id) override {
    conversations_.erase(conversation_id);
    return {};
  }

private:
  std::unordered_map<std::string, std::vector<Message>> conversations_;
};

struct StoredConversation {
  std::size_t version{2};
  std::vector<Message> messages;
};

// Conversation history as one JSON file per conversation id in a directory.
class FileConversationMemory final : public ConversationMemory {
public:
  explicit FileConversationMemory(std::filesystem::path directory)
      : directory_(std::move(directory)) {}

  [[nodiscard]] Result<std::vector<Message>> load(const std::string& conversation_id) override {
    auto stored = read_file(path(conversation_id));
    if (!stored) {
      return std::unexpected(stored.error());
    }
    if (!*stored) {
      return std::vector<Message>{};
    }
    auto conversation = from_json<StoredConversation>(**stored);
    if (!conversation) {
      return std::unexpected(conversation.error());
    }
    if (conversation->version != 2) {
      return std::unexpected(Error{
          .code = ErrorCode::memory,
          .message = "Unsupported stored conversation version: " +
                     std::to_string(conversation->version) + ".",
      });
    }
    for (auto& message : conversation->messages) {
      for (auto& part : message.content) {
        auto valid = std::visit(
            [](auto& value) -> Result<void> {
              if constexpr (requires { value.bytes; }) {
                auto bytes = detail::base64_decode(value.bytes);
                if (!bytes)
                  return std::unexpected(
                      Error{.code = ErrorCode::memory,
                            .message = "Stored attachment contains invalid base64."});
                value.bytes = std::move(*bytes);
              }
              return {};
            },
            part);
        if (!valid)
          return std::unexpected(valid.error());
      }
    }
    return std::move(conversation->messages);
  }

  Result<void> append(const std::string& conversation_id, std::vector<Message> messages) override {
    auto conversation = load(conversation_id);
    if (!conversation) {
      return std::unexpected(conversation.error());
    }
    conversation->insert(conversation->end(), std::make_move_iterator(messages.begin()),
                         std::make_move_iterator(messages.end()));
    return write_file(path(conversation_id),
                      StoredConversation{.messages = std::move(*conversation)});
  }

  Result<void> clear(const std::string& conversation_id) override {
    std::error_code error;
    std::filesystem::remove(path(conversation_id), error);
    if (error) {
      return std::unexpected(Error{.code = ErrorCode::memory, .message = error.message()});
    }
    return {};
  }

private:
  [[nodiscard]] std::filesystem::path path(const std::string& conversation_id) const {
    return directory_ / (detail::sanitize_conversation_id(conversation_id) + ".json");
  }

  [[nodiscard]] static Result<std::optional<std::string>>
  read_file(const std::filesystem::path& file) {
    std::error_code error;
    if (!std::filesystem::exists(file, error)) {
      if (error) {
        return std::unexpected(Error{.code = ErrorCode::memory, .message = error.message()});
      }
      return std::nullopt;
    }
    std::ifstream stream(file, std::ios::binary);
    if (!stream) {
      return std::unexpected(
          Error{.code = ErrorCode::memory, .message = "Could not open " + file.string() + "."});
    }
    return std::string{std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>{}};
  }

  [[nodiscard]] static Result<void> write_file(const std::filesystem::path& file,
                                               StoredConversation conversation) {
    for (auto& message : conversation.messages) {
      for (auto& part : message.content) {
        std::visit(
            [](auto& value) {
              if constexpr (requires { value.bytes; })
                value.bytes = detail::base64_encode(value.bytes);
            },
            part);
      }
    }
    auto json = to_json(conversation);
    if (!json) {
      return std::unexpected(json.error());
    }
    std::error_code error;
    std::filesystem::create_directories(file.parent_path(), error);
    if (error) {
      return std::unexpected(Error{.code = ErrorCode::memory, .message = error.message()});
    }
    const auto temporary = file.string() + ".tmp";
    {
      std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
      if (!stream || !(stream << *json)) {
        return std::unexpected(
            Error{.code = ErrorCode::memory, .message = "Could not write " + temporary + "."});
      }
    }
    std::filesystem::rename(temporary, file, error);
    if (error) {
      return std::unexpected(Error{.code = ErrorCode::memory, .message = error.message()});
    }
    return {};
  }

  std::filesystem::path directory_{};
};

} // namespace cail
