#pragma once

#include <cail/generation.hpp>
#include <cail/language_model.hpp>
#include <cail/memory.hpp>
#include <cail/tool.hpp>

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cail {

struct AgentConfig {
  LanguageModel model;
  std::string instructions;
  std::vector<Tool> tools;
  // Single user messages and text prompts load and store history per conversation id.
  std::shared_ptr<ConversationMemory> memory;
  std::string conversation_id;
};

class Agent {
public:
  explicit Agent(AgentConfig config)
      : model_(std::move(config.model)), instructions_(std::move(config.instructions)),
        tools_(std::move(config.tools)), memory_(std::move(config.memory)),
        conversation_id_(std::move(config.conversation_id)) {}

  [[nodiscard]] Result<GenerationResponse> generate(std::string_view prompt,
                                                    ToolLoopOptions options = {}) const {
    return generate(Message{.content = {TextPart{.text = std::string{prompt}}}},
                    std::move(options));
  }

  [[nodiscard]] Result<GenerationResponse> stream(std::string_view prompt,
                                                  const StreamHandler& on_event,
                                                  ToolLoopOptions options = {}) const {
    return stream(Message{.content = {TextPart{.text = std::string{prompt}}}}, on_event,
                  std::move(options));
  }

  [[nodiscard]] Result<void> generate_async(std::string_view prompt,
                                            LanguageModel::GenerationCompletion complete,
                                            ToolLoopOptions options = {}) const {
    return generate_async(Message{.content = {TextPart{.text = std::string{prompt}}}},
                          std::move(complete), std::move(options));
  }

  // Single user messages use memory when a conversation id resolves:
  // history is loaded, the message is appended, and the new turn is
  // stored after success. A failed call stores nothing.
  [[nodiscard]] Result<GenerationResponse> generate(Message message,
                                                    ToolLoopOptions options = {}) const {
    return message_with_memory(
        std::move(message), std::move(options),
        [this](GenerationRequest request, const ToolLoopOptions& loop_options) {
          return run_tool_loop(model_, prepare(std::move(request)), tools_, loop_options);
        });
  }

  // Explicit requests bypass memory: nothing is loaded and nothing is stored.
  [[nodiscard]] Result<GenerationResponse> generate(GenerationRequest request,
                                                    ToolLoopOptions options = {}) const {
    return run_tool_loop(model_, prepare(std::move(request)), tools_, std::move(options));
  }

  [[nodiscard]] Result<GenerationResponse> stream(Message message, const StreamHandler& on_event,
                                                  ToolLoopOptions options = {}) const {
    return message_with_memory(
        std::move(message), std::move(options),
        [this, &on_event](GenerationRequest request, const ToolLoopOptions& loop_options) {
          return stream_tool_loop(model_, prepare(std::move(request)), tools_, on_event,
                                  loop_options);
        });
  }

  [[nodiscard]] Result<GenerationResponse> stream(GenerationRequest request,
                                                  const StreamHandler& on_event,
                                                  ToolLoopOptions options = {}) const {
    return stream_tool_loop(model_, prepare(std::move(request)), tools_, on_event,
                            std::move(options));
  }

  [[nodiscard]] Result<void> generate_async(GenerationRequest request,
                                            LanguageModel::GenerationCompletion complete,
                                            ToolLoopOptions options = {}) const {
    return run_tool_loop_async(model_, prepare(std::move(request)), tools_, std::move(complete),
                               std::move(options));
  }

  [[nodiscard]] Result<void> generate_async(Message message,
                                            LanguageModel::GenerationCompletion complete,
                                            ToolLoopOptions options = {}) const {
    if (!complete)
      return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                   .message = "Async agents require a completion handler."});
    auto messages = message_history(std::move(message), options);
    if (!messages)
      return std::unexpected(messages.error());
    const auto id = options.conversation_id.empty() ? conversation_id_ : options.conversation_id;
    auto prompt_message = messages->back();
    return generate_async(
        GenerationRequest{.messages = std::move(*messages)},
        [memory = memory_, id, prompt_message = std::move(prompt_message),
         complete = std::move(complete)](Result<GenerationResponse> response) {
          complete(store_response(memory, id, prompt_message, std::move(response)));
        },
        std::move(options));
  }

private:
  [[nodiscard]] Result<std::vector<Message>> message_history(Message message,
                                                             const ToolLoopOptions& options) const {
    if (options.stop.stop_requested())
      return std::unexpected(generation_cancelled_error());
    if (message.role != MessageRole::user || !message.tool_call_id.empty() ||
        !message.tool_calls.empty() || message.content.empty()) {
      return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                   .message = "Agent memory input requires a nonempty user message "
                                              "without tool calls or a tool call ID."});
    }
    const auto& id = options.conversation_id.empty() ? conversation_id_ : options.conversation_id;
    std::vector<Message> messages;
    if (memory_ && !id.empty()) {
      auto history = memory_->load(id);
      if (!history)
        return std::unexpected(history.error());
      if (options.keep_last_messages)
        trim_messages(*history, options.keep_last_messages);
      messages = std::move(*history);
    }
    messages.push_back(std::move(message));
    return messages;
  }

  [[nodiscard]] static Result<GenerationResponse>
  store_response(const std::shared_ptr<ConversationMemory>& memory, const std::string& id,
                 const Message& prompt, Result<GenerationResponse> response) {
    if (response && memory && !id.empty()) {
      std::vector<Message> stored{prompt};
      stored.insert(stored.end(), response->turn.begin(), response->turn.end());
      if (auto appended = memory->append(id, std::move(stored)); !appended)
        return std::unexpected(appended.error());
    }
    return response;
  }

  template <typename Send>
  [[nodiscard]] Result<GenerationResponse>
  message_with_memory(Message message, ToolLoopOptions options, Send&& send) const {
    auto messages = message_history(std::move(message), options);
    if (!messages)
      return std::unexpected(messages.error());
    const auto& id = options.conversation_id.empty() ? conversation_id_ : options.conversation_id;
    auto prompt_message = messages->back();
    return store_response(memory_, id, prompt_message,
                          send(GenerationRequest{.messages = std::move(*messages)}, options));
  }

  [[nodiscard]] GenerationRequest prepare(GenerationRequest request) const {
    if (!instructions_.empty()) {
      request.messages.insert(
          request.messages.begin(),
          Message{.role = MessageRole::system, .content = {TextPart{.text = instructions_}}});
    }
    return request;
  }

  LanguageModel model_;
  std::string instructions_;
  std::vector<Tool> tools_;
  std::shared_ptr<ConversationMemory> memory_;
  std::string conversation_id_;
};

} // namespace cail
