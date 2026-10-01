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
                                                    const ToolLoopOptions& options = {}) const {
    return generate(Message{.content = {TextPart{.text = std::string{prompt}}}}, options);
  }

  [[nodiscard]] Result<GenerationResponse> stream(std::string_view prompt,
                                                  const StreamHandler& on_event,
                                                  const ToolLoopOptions& options = {}) const {
    return stream(Message{.content = {TextPart{.text = std::string{prompt}}}}, on_event, options);
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
                                                    const ToolLoopOptions& options = {}) const {
    return message_with_memory(
        std::move(message), options,
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
                                                  const ToolLoopOptions& options = {}) const {
    return message_with_memory(
        std::move(message), options,
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
    return message_async(std::move(message), {}, std::move(complete), std::move(options));
  }

  [[nodiscard]] Result<void> stream_async(std::string_view prompt, StreamHandler on_event,
                                          LanguageModel::GenerationCompletion complete,
                                          ToolLoopOptions options = {}) const {
    return stream_async(Message{.content = {TextPart{.text = std::string{prompt}}}},
                        std::move(on_event), std::move(complete), std::move(options));
  }

  [[nodiscard]] Result<void> stream_async(Message message, StreamHandler on_event,
                                          LanguageModel::GenerationCompletion complete,
                                          ToolLoopOptions options = {}) const {
    if (!on_event) {
      return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                   .message = "Async streaming requires an event handler."});
    }
    return message_async(std::move(message), std::move(on_event), std::move(complete),
                         std::move(options));
  }

  [[nodiscard]] Result<void> stream_async(GenerationRequest request, StreamHandler on_event,
                                          LanguageModel::GenerationCompletion complete,
                                          ToolLoopOptions options = {}) const {
    return stream_tool_loop_async(model_, prepare(std::move(request)), tools_, std::move(on_event),
                                  std::move(complete), std::move(options));
  }

  [[nodiscard]] Task<Result<GenerationResponse>>
  generate_async(std::string_view prompt, ToolLoopOptions options = {}) const {
    return generate_async(Message{.content = {TextPart{.text = std::string{prompt}}}},
                          std::move(options));
  }

  [[nodiscard]] Task<Result<GenerationResponse>>
  generate_async(Message message, ToolLoopOptions options = {}) const {
    return await_turn(std::move(message), {}, std::move(options));
  }

  [[nodiscard]] Task<Result<GenerationResponse>>
  generate_async(GenerationRequest request, ToolLoopOptions options = {}) const {
    return await_turn(std::move(request), {}, std::move(options));
  }

  [[nodiscard]] Task<Result<GenerationResponse>> stream_async(std::string_view prompt,
                                                              StreamHandler on_event,
                                                              ToolLoopOptions options = {}) const {
    return stream_async(Message{.content = {TextPart{.text = std::string{prompt}}}},
                        std::move(on_event), std::move(options));
  }

  [[nodiscard]] Task<Result<GenerationResponse>>
  stream_async(Message message, StreamHandler on_event, ToolLoopOptions options = {}) const {
    return await_turn(std::move(message), std::move(on_event), std::move(options), true);
  }

  [[nodiscard]] Task<Result<GenerationResponse>> stream_async(GenerationRequest request,
                                                              StreamHandler on_event,
                                                              ToolLoopOptions options = {}) const {
    return await_turn(std::move(request), std::move(on_event), std::move(options), true);
  }

private:
  template <typename Input>
  Task<Result<GenerationResponse>> await_turn(Input input, StreamHandler on_event,
                                              ToolLoopOptions options,
                                              bool streaming = false) const {
    const auto stop = options.stop;
    return detail::await_result<GenerationResponse>(
        [agent = *this, input = std::move(input), on_event = std::move(on_event),
         options = std::move(options),
         streaming](auto complete, const auto& token, auto callbacks) mutable {
          options.stop = std::move(token);
          if (streaming) {
            if (!options.async.schedule) {
              options.async.schedule = std::move(callbacks.schedule);
            }
            return agent.stream_async(std::move(input), std::move(on_event), std::move(complete),
                                      std::move(options));
          }
          return agent.generate_async(std::move(input), std::move(complete), std::move(options));
        },
        stop);
  }

  struct AsyncTurn : std::enable_shared_from_this<AsyncTurn> {
    std::shared_ptr<Agent> agent;
    std::shared_ptr<void> turn;
    Message prompt;
    std::string id;
    ToolLoopOptions options;
    StreamHandler on_event;
    LanguageModel::GenerationCompletion complete;

    void finish(Result<GenerationResponse> result) {
      turn.reset();
      complete(std::move(result));
    }

    void store(Result<GenerationResponse> result) {
      if (!result || result->tool_continuation || !agent->memory_ || id.empty()) {
        finish(std::move(result));
        return;
      }
      if (options.stop.stop_requested()) {
        finish(std::unexpected(generation_cancelled_error()));
        return;
      }
      std::vector<Message> messages{prompt};
      messages.insert(messages.end(), result->turn.begin(), result->turn.end());
      auto self = shared_from_this();
      auto started = detail::initiate_async<void>(
          [self, result = std::move(result)](Result<void> stored) mutable {
            self->finish(stored ? std::move(result)
                                : Result<GenerationResponse>{std::unexpected(stored.error())});
          },
          [&](auto done) {
            return agent->memory_->append_async(id, std::move(messages), std::move(done),
                                                options.stop);
          });
      if (!started) {
        finish(std::unexpected(started.error()));
      }
    }

    Result<void> generate(std::vector<Message> history) {
      if (options.stop.stop_requested()) {
        return std::unexpected(generation_cancelled_error());
      }
      trim_turns(history, options.keep_last_turns);
      history.push_back(prompt);
      auto completion = [self = shared_from_this()](Result<GenerationResponse> result) {
        self->store(std::move(result));
      };
      auto request = agent->prepare(GenerationRequest{.messages = std::move(history)});
      if (on_event) {
        return stream_tool_loop_async(agent->model_, std::move(request), agent->tools_, on_event,
                                      std::move(completion), options);
      }
      return run_tool_loop_async(agent->model_, std::move(request), agent->tools_,
                                 std::move(completion), options);
    }
  };

  [[nodiscard]] Result<void> message_async(Message message, StreamHandler on_event,
                                           LanguageModel::GenerationCompletion complete,
                                           ToolLoopOptions options) const {
    if (!complete) {
      return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                   .message = "Async agents require a completion handler."});
    }
    if (message.role != MessageRole::user || !message.tool_call_id.empty() ||
        !message.tool_calls.empty() || message.content.empty()) {
      return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                   .message = "Agent memory input requires a nonempty user message "
                                              "without tool calls or a tool call ID."});
    }
    auto acquired = acquire_turn(options);
    if (!acquired) {
      return std::unexpected(acquired.error());
    }
    auto state = std::make_shared<AsyncTurn>();
    state->agent = std::make_shared<Agent>(*this);
    state->turn = std::move(*acquired);
    state->prompt = std::move(message);
    state->id = options.conversation_id.empty() ? conversation_id_ : options.conversation_id;
    if (on_event) {
      if (!options.async.max_pending_events) {
        return std::unexpected(
            detail::async_callback_error("max_pending_events must be positive."));
      }
      auto delivery = std::make_shared<detail::StreamDelivery>(
          std::move(on_event), std::move(complete), options.async, options.stop);
      state->on_event = [delivery](const StreamEvent& event) { delivery->event(event); };
      state->complete = [delivery](Result<GenerationResponse> result) {
        delivery->finish(std::move(result));
      };
      options.stop = delivery->token();
    } else {
      state->complete =
          detail::scheduled_completion<GenerationResponse>(std::move(complete), options.async);
    }
    options.async = {};
    state->options = std::move(options);
    if (!memory_ || state->id.empty()) {
      return state->generate({});
    }
    return detail::initiate_async<std::vector<Message>>(
        [state](Result<std::vector<Message>> history) {
          if (!history) {
            state->finish(std::unexpected(history.error()));
          } else if (auto started = state->generate(std::move(*history)); !started) {
            state->finish(std::unexpected(started.error()));
          }
        },
        [&](auto done) {
          return memory_->load_async(state->id, std::move(done), state->options.stop);
        });
  }

  [[nodiscard]] Result<std::shared_ptr<void>> acquire_turn(const ToolLoopOptions& options) const {
    if (options.stop.stop_requested()) {
      return std::unexpected(generation_cancelled_error());
    }
    const auto& id = options.conversation_id.empty() ? conversation_id_ : options.conversation_id;
    if (memory_ && !id.empty()) {
      return memory_->acquire_turn(id);
    }
    return std::shared_ptr<void>{};
  }

  [[nodiscard]] Result<std::vector<Message>> message_history(Message message,
                                                             const ToolLoopOptions& options) const {
    if (options.stop.stop_requested()) {
      return std::unexpected(generation_cancelled_error());
    }
    if (message.role != MessageRole::user || !message.tool_call_id.empty() ||
        !message.tool_calls.empty() || message.content.empty()) {
      return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                   .message = "Agent memory input requires a nonempty user message "
                                              "without tool calls or a tool call ID."});
    }
    const auto& id = options.conversation_id.empty() ? conversation_id_ : options.conversation_id;
    std::vector<Message> messages;
    if (memory_ && !id.empty()) {
      auto history = detail::memory_operation([&] { return memory_->load(id); });
      if (!history) {
        return std::unexpected(history.error());
      }
      trim_turns(*history, options.keep_last_turns);
      messages = std::move(*history);
    }
    messages.push_back(std::move(message));
    return messages;
  }

  [[nodiscard]] static Result<GenerationResponse>
  store_response(const std::shared_ptr<ConversationMemory>& memory, const std::string& id,
                 const Message& prompt, Result<GenerationResponse> response) {
    if (response && !response->tool_continuation && memory && !id.empty()) {
      std::vector<Message> stored{prompt};
      stored.insert(stored.end(), response->turn.begin(), response->turn.end());
      if (auto appended =
              detail::memory_operation([&] { return memory->append(id, std::move(stored)); });
          !appended) {
        return std::unexpected(appended.error());
      }
    }
    return response;
  }

  template <typename Send>
  [[nodiscard]] Result<GenerationResponse>
  message_with_memory(Message message, const ToolLoopOptions& options, Send&& send) const {
    auto turn = acquire_turn(options);
    if (!turn) {
      return std::unexpected(turn.error());
    }
    auto messages = message_history(std::move(message), options);
    if (!messages) {
      return std::unexpected(messages.error());
    }
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
