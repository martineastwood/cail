#pragma once

#include <cail/error.hpp>
#include <cail/generation.hpp>
#include <cail/json.hpp>
#include <cail/language_model.hpp>

#include <algorithm>
#include <concepts>
#include <condition_variable>
#include <cstddef>
#include <deque>
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
  using Completion = std::function<void(Result<std::string>)>;
  using AsyncExecutor = std::function<void(ToolCall, ToolContext, Completion)>;
  using Executor = std::function<Result<std::string>(const ToolCall&, const ToolContext&)>;

  Tool(ToolDefinition definition, Executor executor)
      : definition_(std::move(definition)), executor_(std::move(executor)) {}

  Tool(ToolDefinition definition, AsyncExecutor executor)
      : definition_(std::move(definition)), async_executor_(std::move(executor)) {}

  [[nodiscard]] const ToolDefinition& definition() const noexcept { return definition_; }

  [[nodiscard]] Result<std::string> execute(const ToolCall& call,
                                            const ToolContext& context) const {
    try {
      if (executor_)
        return executor_(call, context);
      struct Pending {
        std::mutex mutex;
        std::condition_variable ready;
        std::optional<Result<std::string>> result;
      };
      auto pending = std::make_shared<Pending>();
      std::stop_callback cancelled(context.stop, [pending] {
        std::lock_guard lock(pending->mutex);
        pending->ready.notify_all();
      });
      async_executor_(call, context, [pending](Result<std::string> result) {
        std::lock_guard lock(pending->mutex);
        if (!pending->result)
          pending->result = std::move(result);
        pending->ready.notify_all();
      });
      std::unique_lock lock(pending->mutex);
      pending->ready.wait(
          lock, [&] { return pending->result.has_value() || context.stop.stop_requested(); });
      if (context.stop.stop_requested())
        return std::unexpected(generation_cancelled_error());
      return std::move(*pending->result);
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

  // Async handlers resume directly; synchronous handlers use bounded workers.
  [[nodiscard]] Result<void> execute_async(ToolCall call, ToolContext context,
                                           Completion complete) const {
    if (!complete)
      return std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                   .message = "Async tools require a completion handler."});
    auto done = detail::complete_once<std::string>(std::move(complete));
    if (context.stop.stop_requested()) {
      done(std::unexpected(generation_cancelled_error()));
      return {};
    }
    if (executor_) {
      const auto stop = context.stop;
      return detail::tool_workers().post(
          [tool = *this, call = std::move(call), context = std::move(context),
           done](std::stop_token worker_stop) mutable {
            detail::LinkedStop linked(context.stop, worker_stop);
            context.stop = linked.source.get_token();
            if (context.stop.stop_requested())
              done(std::unexpected(generation_cancelled_error()));
            else
              done(tool.execute(call, context));
          },
          stop);
    }
    struct Pending {
      Pending(Tool tool, Completion complete)
          : tool(std::move(tool)), complete(std::move(complete)) {}
      Tool tool;
      Completion complete;
      std::optional<std::stop_callback<std::function<void()>>> cancel;
    };
    auto pending = std::make_shared<Pending>(*this, done);
    pending->cancel.emplace(context.stop, [weak = std::weak_ptr(pending)] {
      if (auto active = weak.lock())
        active->complete(std::unexpected(generation_cancelled_error()));
    });
    if (context.stop.stop_requested())
      return {};
    try {
      pending->tool.async_executor_(
          std::move(call), std::move(context),
          [pending](Result<std::string> result) { pending->complete(std::move(result)); });
    } catch (const std::exception& error) {
      done(std::unexpected(
          Error{.code = ErrorCode::tool_execution,
                .message = "Tool execution failed for " + definition_.name + ": " + error.what()}));
    } catch (...) {
      done(
          std::unexpected(Error{.code = ErrorCode::tool_execution,
                                .message = "Tool execution failed for " + definition_.name + "."}));
    }
    return {};
  }

private:
  ToolDefinition definition_;
  Executor executor_;
  AsyncExecutor async_executor_;
};

struct ToolLoopOptions {
  std::size_t max_rounds{8};
  AsyncOptions async;
  std::function<bool(const GenerationResponse&, const MiddlewareContext&)> stop_when;
  std::stop_token stop;
  // Conversation id for agent memory; overrides the agent's default.
  std::string conversation_id;
  // Send this many previous user turns, including their assistant/tool exchanges.
  // Leading system/developer messages stay. Zero sends the full history.
  std::size_t keep_last_turns{0};
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

  ExecutableTool(std::string name, std::string description, Tool::AsyncExecutor executor)
      : Tool(make_tool<Input>(std::move(name), std::move(description)), std::move(executor)) {}

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

// The handler owns its work and calls completion exactly once, possibly inline.
template <typename Input, typename Output, typename Handler>
[[nodiscard]] ExecutableTool<Input, Output> async_tool(std::string name, std::string description,
                                                       Handler&& handler) {
  auto shared_handler = std::make_shared<std::decay_t<Handler>>(std::forward<Handler>(handler));
  return ExecutableTool<Input, Output>(
      std::move(name), std::move(description),
      Tool::AsyncExecutor{
          [shared_handler](ToolCall call, ToolContext context, Tool::Completion complete) {
            auto input = from_json<Input>(call.arguments);
            if (!input) {
              complete(std::unexpected(input.error()));
              return;
            }
            std::invoke(*shared_handler, std::move(*input), std::move(context),
                        [complete = std::move(complete)](Result<Output> output) {
                          complete(output ? to_json(*output)
                                          : Result<std::string>{std::unexpected(output.error())});
                        });
          }});
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

  [[nodiscard]] Result<bool> begin_step(Result<GenerationResponse> response) {
    if (response) {
      result = *response;
      turn.push_back(Message{
          .role = MessageRole::assistant,
          .content = response->text.empty()
                         ? std::vector<ContentPart>{}
                         : std::vector<ContentPart>{TextPart{.text = response->text}},
          .tool_calls = response->tool_calls,
          .provider_options = response->provider_options,
      });
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
    }
    if (auto middleware = run_after_step(request, response); !middleware)
      return std::unexpected(middleware.error());
    if (options.stop.stop_requested())
      return std::unexpected(generation_cancelled_error());
    if (!response)
      return std::unexpected(response.error());
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
    follow_up = GenerationRequest{
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
    calls = std::move(response->tool_calls);
    tool_index = 0;
    return false;
  }

  [[nodiscard]] Result<const Tool*> next_tool() const {
    if (options.stop.stop_requested())
      return std::unexpected(generation_cancelled_error());
    const auto& call = calls[tool_index];
    const auto tool = std::ranges::find_if(
        tools, [&call](const auto& candidate) { return candidate.definition().name == call.name; });
    if (tool == tools.end())
      return std::unexpected(
          Error{.code = ErrorCode::tool_not_found,
                .message = "The model requested an unregistered tool: " + call.name});
    return &*tool;
  }

  [[nodiscard]] Result<void> accept_tool(Result<std::string> output) {
    if (!output)
      return std::unexpected(output.error());
    const auto& call = calls[tool_index];
    tool_results.push_back({.call_id = call.id, .name = call.name, .output = *output});
    Message message{.role = MessageRole::tool,
                    .content = {TextPart{.text = std::move(*output)}},
                    .tool_call_id = call.id};
    follow_up.messages.push_back(message);
    turn.push_back(std::move(message));
    ++tool_index;
    if (options.stop.stop_requested())
      return std::unexpected(generation_cancelled_error());
    return {};
  }

  void advance() {
    ++round;
    request = std::move(follow_up);
  }

  [[nodiscard]] Result<bool> accept(Result<GenerationResponse> response) {
    auto finished = begin_step(std::move(response));
    if (!finished || *finished)
      return finished;
    while (tool_index < calls.size()) {
      auto tool = next_tool();
      if (!tool)
        return std::unexpected(tool.error());
      const auto& call = calls[tool_index];
      if (auto output = accept_tool((*tool)->execute(call, tool_context())); !output)
        return std::unexpected(output.error());
    }
    advance();
    return false;
  }

  [[nodiscard]] Error with_progress(Error error) const {
    if (!steps.empty() || !turn.empty()) {
      auto partial = result;
      partial.steps = steps;
      partial.total_usage = total_usage;
      partial.tool_results = tool_results;
      partial.turn = turn;
      error.partial_response = std::make_shared<GenerationResponse>(std::move(partial));
    }
    return error;
  }

  ToolContext tool_context() const {
    return {.call_id = calls[tool_index].id, .round = round, .stop = options.stop};
  }

  std::vector<ToolCall> calls;
  std::size_t tool_index{};

  GenerationRequest request;
  ToolLoopOptions options;
  GenerationResponse result;

private:
  GenerationRequest follow_up;
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
      return std::unexpected(loop.with_progress(generation_cancelled_error()));
    auto finished = loop.accept(send(loop.request));
    if (!finished)
      return std::unexpected(loop.with_progress(finished.error()));
    if (*finished)
      return std::move(loop.result);
  }
}

class AsyncToolLoop : public std::enable_shared_from_this<AsyncToolLoop> {
public:
  AsyncToolLoop(LanguageModel model, GenerationRequest request, std::vector<Tool> tools,
                ToolLoopOptions options, LanguageModel::GenerationCompletion complete,
                StreamHandler on_event = {})
      : loop_(std::move(request), std::move(tools), std::move(options)), model_(std::move(model)),
        complete_(complete_once<GenerationResponse>(std::move(complete))),
        on_event_(std::move(on_event)) {}

  [[nodiscard]] Result<void> start() {
    auto completion = [self = shared_from_this()](Result<GenerationResponse> response) {
      self->dispatch([self, response = std::move(response)]() mutable {
        auto finished = self->loop_.begin_step(std::move(response));
        if (!finished)
          self->finish(std::unexpected(finished.error()));
        else if (*finished)
          self->finish(std::move(self->loop_.result));
        else
          self->next_tool();
      });
    };
    if (on_event_)
      return model_.stream_async(loop_.request, on_event_, std::move(completion),
                                 loop_.options.stop);
    return model_.generate_async(loop_.request, std::move(completion), loop_.options.stop);
  }

private:
  // One operation has one serial trampoline, including inline tool completions.
  // No worker is occupied while an asynchronous provider or tool is pending.
  void dispatch(std::function<void()> task) {
    std::unique_lock lock(mutex_);
    pending_.push_back(std::move(task));
    if (running_)
      return;
    running_ = true;
    while (!pending_.empty()) {
      auto next = std::move(pending_.front());
      pending_.pop_front();
      lock.unlock();
      if (!finished_) {
        try {
          next();
        } catch (const std::exception& error) {
          finish(std::unexpected(
              Error{.code = ErrorCode::invalid_configuration,
                    .message = std::string{"Async generation failed: "} + error.what()}));
        } catch (...) {
          finish(std::unexpected(Error{.code = ErrorCode::invalid_configuration,
                                       .message = "Async generation failed."}));
        }
      }
      lock.lock();
    }
    running_ = false;
  }

  void next_tool() {
    if (loop_.tool_index == loop_.calls.size()) {
      loop_.advance();
      if (auto started = start(); !started)
        finish(std::unexpected(started.error()));
      return;
    }
    auto tool = loop_.next_tool();
    if (!tool) {
      finish(std::unexpected(tool.error()));
      return;
    }
    auto started = (*tool)->execute_async(
        loop_.calls[loop_.tool_index], loop_.tool_context(),
        [self = shared_from_this()](Result<std::string> output) {
          self->dispatch([self, output = std::move(output)]() mutable {
            if (auto accepted = self->loop_.accept_tool(std::move(output)); !accepted)
              self->finish(std::unexpected(accepted.error()));
            else
              self->next_tool();
          });
        });
    if (!started)
      finish(std::unexpected(started.error()));
  }

  void finish(Result<GenerationResponse> response) {
    if (!response)
      response = std::unexpected(loop_.with_progress(response.error()));
    finished_ = true;
    complete_(std::move(response));
  }

  ToolLoop loop_;
  LanguageModel model_;
  LanguageModel::GenerationCompletion complete_;
  StreamHandler on_event_;
  std::mutex mutex_;
  bool running_{};
  bool finished_{};
  std::deque<std::function<void()>> pending_;
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
  auto done = detail::scheduled_completion<GenerationResponse>(std::move(complete), options.async);
  return std::make_shared<detail::AsyncToolLoop>(std::move(model), std::move(request),
                                                 std::move(tools), std::move(options),
                                                 std::move(done))
      ->start();
}

[[nodiscard]] inline Result<void>
stream_tool_loop_async(LanguageModel model, GenerationRequest request, std::vector<Tool> tools,
                       StreamHandler on_event, LanguageModel::GenerationCompletion complete,
                       ToolLoopOptions options = {}) {
  if (!on_event || !complete)
    return std::unexpected(
        Error{.code = ErrorCode::invalid_configuration,
              .message = "Async streaming requires event and completion handlers."});
  if (options.stop.stop_requested())
    return std::unexpected(generation_cancelled_error());
  if (!options.async.max_pending_events)
    return std::unexpected(detail::async_callback_error("max_pending_events must be positive."));
  auto delivery = std::make_shared<detail::StreamDelivery>(std::move(on_event), std::move(complete),
                                                           options.async, options.stop);
  options.stop = delivery->token();
  options.async = {};
  auto started =
      std::make_shared<detail::AsyncToolLoop>(
          std::move(model), std::move(request), std::move(tools), std::move(options),
          [delivery](Result<GenerationResponse> result) { delivery->finish(std::move(result)); },
          [delivery](const StreamEvent& event) { delivery->event(event); })
          ->start();
  if (!started)
    delivery->abort();
  return started;
}

} // namespace cail
