#pragma once

#include <cail/error.hpp>
#include <cail/generation.hpp>
#include <cail/json.hpp>
#include <cail/language_model.hpp>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <exception>
#include <expected>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace cail {

struct ToolContext {
  std::string call_id;
  std::size_t round{};
  std::stop_token stop;
};

class Tool {
public:
  using Executor = std::function<Result<std::string>(const ToolCall&, const ToolContext&)>;

  Tool(ToolDefinition definition, Executor executor)
      : definition_(std::move(definition)), executor_(std::move(executor)) {}

  [[nodiscard]] const ToolDefinition& definition() const noexcept { return definition_; }

  [[nodiscard]] Result<std::string> execute(const ToolCall& call,
                                            const ToolContext& context) const {
    try {
      return executor_(call, context);
    } catch (const std::exception& error) {
      return std::unexpected(Error{
          .code = ErrorCode::tool_execution,
          .message = "Tool execution failed for " + definition_.name + ": " + error.what(),
      });
    } catch (...) {
      return std::unexpected(Error{
          .code = ErrorCode::tool_execution,
          .message = "Tool execution failed for " + definition_.name + ".",
      });
    }
  }

private:
  ToolDefinition definition_;
  Executor executor_;
};

struct ToolLoopOptions {
  std::size_t max_rounds{8};
  std::function<bool(const GenerationResponse&, const MiddlewareContext&)> stop_when;
  std::stop_token stop;
  // Conversation id for agent memory; overrides the agent's default.
  std::string conversation_id;
  // Send at most this many trailing history messages (leading system and
  // developer messages are always kept). Zero sends the full history.
  std::size_t keep_last_messages{0};
};

namespace detail {

template <typename T> struct tool_return_traits {
  using value_type = T;
  static constexpr bool is_result = false;
};

template <typename T> struct tool_return_traits<std::expected<T, Error>> {
  using value_type = T;
  static constexpr bool is_result = true;
};

template <typename Handler, typename Input>
decltype(auto) invoke_tool(Handler& handler, const Input& input, const ToolContext& context) {
  if constexpr (std::invocable<Handler&, const Input&, const ToolContext&>) {
    return std::invoke(handler, input, context);
  } else {
    return std::invoke(handler, input);
  }
}

template <typename Input, typename Output, typename Handler>
[[nodiscard]] Tool::Executor make_tool_executor(Handler&& handler) {
  using HandlerType = std::decay_t<Handler>;
  using HandlerReturn = std::remove_cvref_t<decltype(invoke_tool(
      std::declval<HandlerType&>(), std::declval<const Input&>(),
      std::declval<const ToolContext&>()))>;
  using ReturnTraits = tool_return_traits<HandlerReturn>;
  static_assert(
      std::same_as<typename ReturnTraits::value_type, Output>,
      "A CAIL tool handler must return its declared output type or cail::Result of that type.");

  auto shared_handler = std::make_shared<HandlerType>(std::forward<Handler>(handler));
  return [shared_handler](const ToolCall& call, const ToolContext& context) -> Result<std::string> {
    const auto arguments = from_json<Input>(call.arguments);
    if (!arguments) {
      return std::unexpected(arguments.error());
    }

    auto output = invoke_tool(*shared_handler, *arguments, context);
    if constexpr (ReturnTraits::is_result) {
      if (!output) {
        return std::unexpected(output.error());
      }
      return to_json(*output);
    } else {
      return to_json(output);
    }
  };
}

} // namespace detail

template <typename Input, typename Output> class ExecutableTool : public Tool {
public:
  template <typename Handler>
  ExecutableTool(std::string name, std::string description, Handler&& handler)
      : Tool(make_tool<Input>(std::move(name), std::move(description)),
             detail::make_tool_executor<Input, Output>(std::forward<Handler>(handler))) {}

  [[nodiscard]] Result<Output> decode_output(const ToolResult& result) const {
    if (result.name != definition().name) {
      return std::unexpected(Error{
          .code = ErrorCode::invalid_tool_call,
          .message = "Tool result belongs to " + result.name + ", not " + definition().name + ".",
      });
    }
    return from_json<Output>(result.output);
  }
};

template <typename Input, typename Output, typename Handler>
[[nodiscard]] ExecutableTool<Input, Output> tool(std::string name, std::string description,
                                                 Handler&& handler) {
  return ExecutableTool<Input, Output>{std::move(name), std::move(description),
                                       std::forward<Handler>(handler)};
}

namespace detail {

class ToolLoop {
public:
  ToolLoop(GenerationRequest initial, std::vector<Tool> registered, ToolLoopOptions settings)
      : request(std::move(initial)), options(std::move(settings)), tools(std::move(registered)) {
    request.tools.clear();
    for (const auto& tool : tools)
      request.tools.push_back(tool.definition());
  }

  [[nodiscard]] Result<bool> accept(Result<GenerationResponse> response) {
    if (auto middleware = run_after_step(request, response); !middleware)
      return std::unexpected(middleware.error());
    if (options.stop.stop_requested())
      return std::unexpected(generation_cancelled_error());
    if (!response)
      return std::unexpected(response.error());
    steps.push_back(GenerationStep{.step = request.step,
                                   .text = response->text,
                                   .reasoning = response->reasoning,
                                   .finish_reason = response->finish_reason,
                                   .raw_finish_reason = response->raw_finish_reason,
                                   .usage = response->usage,
                                   .tool_calls = response->tool_calls});
    if (response->usage) {
      if (!total_usage)
        total_usage.emplace();
      add_usage(*total_usage, *response->usage);
    }
    bool stopped = false;
    if (options.stop_when) {
      try {
        stopped = options.stop_when(*response, {.step = request.step});
      } catch (const std::exception& error) {
        return std::unexpected(
            Error{.code = ErrorCode::invalid_configuration,
                  .message = std::string{"The tool-loop stop condition failed: "} + error.what()});
      } catch (...) {
        return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                     .message = "The tool-loop stop condition failed."});
      }
    }
    if (options.stop.stop_requested())
      return std::unexpected(generation_cancelled_error());
    auto assistant = Message{
        .role = MessageRole::assistant,
        .content = response->text.empty()
                       ? std::vector<ContentPart>{}
                       : std::vector<ContentPart>{TextPart{.text = response->text}},
        .tool_calls = response->tool_calls,
        .provider_options = response->provider_options,
    };
    if (stopped || response->status != GenerationStatus::completed ||
        response->tool_calls.empty()) {
      turn.push_back(std::move(assistant));
      response->steps = std::move(steps);
      response->total_usage = total_usage;
      response->turn = std::move(turn);
      response->tool_results = std::move(tool_results);
      result = std::move(*response);
      return true;
    }
    if (round >= options.max_rounds)
      return std::unexpected(
          Error{.code = ErrorCode::tool_loop_limit,
                .message = "The model exceeded the configured tool-call round limit."});
    GenerationRequest follow_up{
        .tools = request.tools,
        .structured_output = request.structured_output,
        .continuation_token = response->continuation_token,
        .session_id = request.session_id,
        .max_output_tokens = request.max_output_tokens,
        .temperature = request.temperature,
        .top_p = request.top_p,
        .stop_sequences = request.stop_sequences,
        .tool_choice =
            request.tool_choice && (request.tool_choice->mode == ToolChoiceMode::required ||
                                    request.tool_choice->mode == ToolChoiceMode::named)
                ? std::optional<ToolChoice>{ToolChoice{}}
                : request.tool_choice,
        .stream_usage = request.stream_usage,
        .provider_options = request.provider_options,
        .middleware = request.middleware,
        .step = request.step + 1,
    };
    if (!response->continuation_token) {
      follow_up.messages = request.messages;
      follow_up.messages.push_back(assistant);
    }
    turn.push_back(std::move(assistant));
    for (const auto& call : response->tool_calls) {
      if (options.stop.stop_requested())
        return std::unexpected(generation_cancelled_error());
      const auto tool = std::ranges::find_if(tools, [&call](const auto& candidate) {
        return candidate.definition().name == call.name;
      });
      if (tool == tools.end())
        return std::unexpected(
            Error{.code = ErrorCode::tool_not_found,
                  .message = "The model requested an unregistered tool: " + call.name});
      auto output = tool->execute(call, {.call_id = call.id, .round = round, .stop = options.stop});
      if (options.stop.stop_requested())
        return std::unexpected(generation_cancelled_error());
      if (!output)
        return std::unexpected(output.error());
      tool_results.push_back({.call_id = call.id, .name = call.name, .output = *output});
      Message tool_message{.role = MessageRole::tool,
                           .content = {TextPart{.text = std::move(*output)}},
                           .tool_call_id = call.id};
      follow_up.messages.push_back(tool_message);
      turn.push_back(std::move(tool_message));
    }
    ++round;
    request = std::move(follow_up);
    return false;
  }

  GenerationRequest request;
  ToolLoopOptions options;
  GenerationResponse result;

private:
  std::vector<Tool> tools;
  std::vector<GenerationStep> steps;
  std::optional<TokenUsage> total_usage;
  std::vector<ToolResult> tool_results;
  std::vector<Message> turn;
  std::size_t round{};
};

template <typename Send>
[[nodiscard]] Result<GenerationResponse> run_tool_loop(GenerationRequest request,
                                                       const std::vector<Tool>& tools,
                                                       ToolLoopOptions options, Send&& send) {
  ToolLoop loop(std::move(request), tools, std::move(options));
  while (true) {
    if (loop.options.stop.stop_requested())
      return std::unexpected(generation_cancelled_error());
    auto finished = loop.accept(send(loop.request));
    if (!finished)
      return std::unexpected(finished.error());
    if (*finished)
      return std::move(loop.result);
  }
}

class AsyncToolLoop : public std::enable_shared_from_this<AsyncToolLoop> {
public:
  AsyncToolLoop(LanguageModel model, GenerationRequest request, std::vector<Tool> tools,
                ToolLoopOptions options, LanguageModel::GenerationCompletion complete)
      : loop_(std::move(request), std::move(tools), std::move(options)), model_(std::move(model)),
        complete_(std::move(complete)) {}

  [[nodiscard]] Result<void> start() {
    return model_.generate_async(
        loop_.request,
        [self = shared_from_this()](Result<GenerationResponse> response) {
          self->accept(std::move(response));
        },
        loop_.options.stop);
  }

private:
  // Inline completions queue the next result instead of recursively growing the stack.
  void accept(Result<GenerationResponse> response) {
    std::unique_lock lock(mutex_);
    pending_ = std::move(response);
    if (running_)
      return;
    running_ = true;
    while (pending_) {
      auto next = std::move(*pending_);
      pending_.reset();
      lock.unlock();
      auto finished = loop_.accept(std::move(next));
      if (!finished) {
        complete_(std::unexpected(finished.error()));
        return;
      }
      if (*finished) {
        complete_(std::move(loop_.result));
        return;
      }
      auto started = start();
      if (!started) {
        complete_(std::unexpected(started.error()));
        return;
      }
      lock.lock();
    }
    running_ = false;
  }

  ToolLoop loop_;
  LanguageModel model_;
  LanguageModel::GenerationCompletion complete_;
  std::mutex mutex_;
  bool running_{};
  std::optional<Result<GenerationResponse>> pending_;
};

} // namespace detail

template <typename Client>
[[nodiscard]] Result<GenerationResponse>
run_tool_loop(const Client& client, GenerationRequest request, const std::vector<Tool>& tools,
              ToolLoopOptions options = {}) {
  return detail::run_tool_loop(
      std::move(request), tools, options,
      [&](const GenerationRequest& step) { return client.generate(step, options.stop); });
}

template <typename Client>
[[nodiscard]] Result<GenerationResponse>
stream_tool_loop(const Client& client, GenerationRequest request, const std::vector<Tool>& tools,
                 const StreamHandler& on_event, ToolLoopOptions options = {}) {
  return detail::run_tool_loop(
      std::move(request), tools, options,
      [&](const GenerationRequest& step) { return client.stream(step, on_event, options.stop); });
}

[[nodiscard]] inline Result<void>
run_tool_loop_async(LanguageModel model, GenerationRequest request, std::vector<Tool> tools,
                    LanguageModel::GenerationCompletion complete, ToolLoopOptions options = {}) {
  if (!complete)
    return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                 .message = "Async tool loops require a completion handler."});
  if (options.stop.stop_requested())
    return std::unexpected(generation_cancelled_error());
  return std::make_shared<detail::AsyncToolLoop>(std::move(model), std::move(request),
                                                 std::move(tools), std::move(options),
                                                 std::move(complete))
      ->start();
}

} // namespace cail
