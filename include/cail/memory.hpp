#pragma once

#include <cail/async.hpp>
#include <cail/detail/base64.hpp>
#include <cail/error.hpp>
#include <cail/generation.hpp>
#include <cail/json.hpp>

#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cail {

// Keep the newest user turns, including all following assistant and tool
// messages, while preserving leading system and developer messages.
// Zero keeps the full history.
inline void trim_turns(std::vector<Message>& messages, std::size_t keep_last) {
  if (!keep_last) {
    return;
  }
  auto prefix = messages.begin();
  while (prefix != messages.end() &&
         (prefix->role == MessageRole::system || prefix->role == MessageRole::developer)) {
    ++prefix;
  }
  std::size_t turns = 0;
  for (auto cursor = messages.end(); cursor != prefix;) {
    --cursor;
    if (cursor->role == MessageRole::user && ++turns == keep_last) {
      messages.erase(prefix, cursor);
      return;
    }
  }
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

template <typename Operation> auto memory_operation(Operation operation) -> decltype(operation()) {
  try {
    return operation();
  } catch (const std::exception& error) {
    return std::unexpected(
        Error{.code = ErrorCode::memory,
              .message = std::string{"Conversation memory failed: "} + error.what()});
  } catch (...) {
    return std::unexpected(
        Error{.code = ErrorCode::memory, .message = "Conversation memory failed."});
  }
}

} // namespace detail

// Storage for agent conversation history. Implement this over your own store
// for durable sessions. Operations must support concurrent calls.
class ConversationMemory : public std::enable_shared_from_this<ConversationMemory> {
public:
  virtual ~ConversationMemory() = default;

  [[nodiscard]] virtual Result<std::vector<Message>> load(const std::string& conversation_id) = 0;

  [[nodiscard]] virtual Result<void> append(const std::string& conversation_id,
                                            std::vector<Message> messages) = 0;

  [[nodiscard]] virtual Result<void> clear(const std::string& conversation_id) = 0;

  using LoadCompletion = std::function<void(Result<std::vector<Message>>)>;
  using Completion = std::function<void(Result<void>)>;

  // Blocking stores use bounded workers. Remote stores can override these methods.
  // Overrides may retain the stop token, so keep the ownership-taking signatures.
  // NOLINTBEGIN(performance-unnecessary-value-param)
  [[nodiscard]] virtual Result<void> load_async(std::string id, LoadCompletion complete,
                                                std::stop_token stop = {}) {
    return post_memory<std::vector<Message>>(
        std::move(complete), stop,
        [id = std::move(id)](ConversationMemory& memory) { return memory.load(id); });
  }

  [[nodiscard]] virtual Result<void> append_async(std::string id, std::vector<Message> messages,
                                                  Completion complete, std::stop_token stop = {}) {
    return post_memory<void>(
        std::move(complete), stop,
        [id = std::move(id), messages = std::move(messages)](ConversationMemory& memory) mutable {
          return memory.append(id, std::move(messages));
        });
  }

  [[nodiscard]] virtual Result<void> clear_async(std::string id, Completion complete,
                                                 std::stop_token stop = {}) {
    return post_memory<void>(
        std::move(complete), stop,
        [id = std::move(id)](ConversationMemory& memory) { return memory.clear(id); });
  }

  // NOLINTEND(performance-unnecessary-value-param)
private:
  template <typename T, typename Callback, typename Work>
  Result<void> post_memory(Callback complete, const std::stop_token& stop, Work work) {
    if (!complete) {
      return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                   .message = "Async memory requires a completion handler."});
    }
    if (stop.stop_requested()) {
      return std::unexpected(generation_cancelled_error());
    }
    auto owner = weak_from_this().lock();
    if (!owner) {
      return std::unexpected(
          Error{.code = ErrorCode::memory, .message = "Async memory requires shared ownership."});
    }
    return detail::memory_workers().post(
        [owner, work = std::move(work), stop, done = detail::complete_once<T>(std::move(complete))](
            const std::stop_token& worker_stop) mutable {
          detail::LinkedStop linked(stop, worker_stop);
          if (linked.source.stop_requested()) {
            done(std::unexpected(generation_cancelled_error()));
          } else {
            auto result = detail::memory_operation([&] { return work(*owner); });
            done(linked.source.stop_requested()
                     ? Result<T>{std::unexpected(generation_cancelled_error())}
                     : std::move(result));
          }
        },
        stop);
  }

  friend class Agent;

  // A complete turn must use one history snapshot. Overlapping turns for the
  // same conversation are rejected instead of silently reordering history.
  [[nodiscard]] Result<std::shared_ptr<void>> acquire_turn(const std::string& id) {
    auto owner = weak_from_this().lock();
    if (!owner) {
      return std::unexpected(
          Error{.code = ErrorCode::memory,
                .message = "Conversation turns require shared memory ownership."});
    }
    std::lock_guard lock(turn_mutex_);
    if (!active_turns_.insert(id).second) {
      return std::unexpected(Error{.code = ErrorCode::memory,
                                   .message = "A turn is already active for conversation: " + id});
    }
    return std::shared_ptr<void>(this, [owner = std::move(owner), id](void*) {
      std::lock_guard lock(owner->turn_mutex_);
      owner->active_turns_.erase(id);
    });
  }

  std::mutex turn_mutex_;
  std::unordered_set<std::string> active_turns_;
};

// Conversation history in process memory. Forgets everything on restart.
class InMemoryConversationMemory final : public ConversationMemory {
public:
  [[nodiscard]] Result<std::vector<Message>> load(const std::string& conversation_id) override {
    std::lock_guard lock(mutex_);
    if (const auto found = conversations_.find(conversation_id); found != conversations_.end()) {
      return found->second;
    }
    return std::vector<Message>{};
  }

  Result<void> append(const std::string& conversation_id, std::vector<Message> messages) override {
    std::lock_guard lock(mutex_);
    auto& conversation = conversations_[conversation_id];
    conversation.insert(conversation.end(), std::make_move_iterator(messages.begin()),
                        std::make_move_iterator(messages.end()));
    return {};
  }

  Result<void> clear(const std::string& conversation_id) override {
    std::lock_guard lock(mutex_);
    conversations_.erase(conversation_id);
    return {};
  }

private:
  std::mutex mutex_;
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
    std::lock_guard lock(mutex_);
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
                if (!bytes) {
                  return std::unexpected(
                      Error{.code = ErrorCode::memory,
                            .message = "Stored attachment contains invalid base64."});
                }
                value.bytes = std::move(*bytes);
              }
              return {};
            },
            part);
        if (!valid) {
          return std::unexpected(valid.error());
        }
      }
    }
    return std::move(conversation->messages);
  }

  Result<void> append(const std::string& conversation_id, std::vector<Message> messages) override {
    std::lock_guard lock(mutex_);
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
    std::lock_guard lock(mutex_);
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
              if constexpr (requires { value.bytes; }) {
                value.bytes = detail::base64_encode(value.bytes);
              }
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

  // File instances in this process share a lock, including read-modify-write
  // and the temporary file used by atomic replacement.
  inline static std::recursive_mutex mutex_;
  std::filesystem::path directory_;
};

} // namespace cail
