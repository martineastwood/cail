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
  // When set, prompt calls load and store history per conversation id.
  std::shared_ptr<ConversationMemory> memory;
  std::string conversation_id;
};

class Agent {
public:
  explicit Agent(AgentConfig config)
      : model_(std::move(config.model)), instructions_(std::move(config.instructions)),
        tools_(std::move(config.tools)), memory_(std::move(config.memory)),
        conversation_id_(std::move(config.conversation_id)) {}

  // Prompt calls use the agent memory when it is configured and a conversation
  // id resolves: history is loaded, the prompt is appended, and the new turn is
  // stored after success. A failed call stores nothing.
  [[nodiscard]] Result<GenerationResponse> generate(std::string_view prompt,
                                                    ToolLoopOptions options = {}) const {
    return prompt_with_memory(
        prompt, std::move(options),
        [this](GenerationRequest request, const ToolLoopOptions& loop_options) {
          return run_tool_loop(model_, prepare(std::move(request)), tools_, loop_options);
        });
  }

  // Explicit requests bypass memory: nothing is loaded and nothing is stored.
  [[nodiscard]] Result<GenerationResponse> generate(GenerationRequest request,
                                                    ToolLoopOptions options = {}) const {
    return run_tool_loop(model_, prepare(std::move(request)), tools_, std::move(options));
  }

  [[nodiscard]] Result<GenerationResponse> stream(std::string_view prompt,
                                                  const StreamHandler& on_event,
                                                  ToolLoopOptions options = {}) const {
    return prompt_with_memory(
        prompt, std::move(options),
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

private:
  template <typename Send>
  [[nodiscard]] Result<GenerationResponse>
  prompt_with_memory(std::string_view prompt, ToolLoopOptions options, Send&& send) const {
    const auto& id = options.conversation_id.empty() ? conversation_id_ : options.conversation_id;
    const bool remembered = memory_ && !id.empty();
    const Message prompt_message{.content = {TextPart{.text = std::string{prompt}}}};
    std::vector<Message> messages;
    if (remembered) {
      auto history = memory_->load(id);
      if (!history) {
        return std::unexpected(history.error());
      }
      if (options.keep_last_messages) {
        trim_messages(*history, options.keep_last_messages);
      }
      messages = std::move(*history);
    }
    messages.push_back(prompt_message);
    auto response = send(GenerationRequest{.messages = std::move(messages)}, options);
    if (!response) {
      return std::unexpected(response.error());
    }
    if (remembered) {
      std::vector<Message> stored{prompt_message};
      stored.insert(stored.end(), response->turn.begin(), response->turn.end());
      if (const auto appended = memory_->append(id, std::move(stored)); !appended) {
        return std::unexpected(appended.error());
      }
    }
    return response;
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
