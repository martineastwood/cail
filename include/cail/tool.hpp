#pragma once

#include <cail/error.hpp>
#include <cail/generation.hpp>
#include <cail/json.hpp>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <exception>
#include <expected>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace cail {

struct ToolContext {
    std::string call_id;
    std::size_t round{};
};

class Tool {
    public:
    using Executor = std::function<Result<std::string>(const ToolCall&, const ToolContext&)>;

    Tool(ToolDefinition definition, Executor executor)
        : definition_(std::move(definition)), executor_(std::move(executor)) {}

    [[nodiscard]] const ToolDefinition& definition() const noexcept { return definition_; }

    [[nodiscard]] Result<std::string> execute(const ToolCall& call, const ToolContext& context) const {
        if (!executor_) {
            return std::unexpected(Error{
                .code = ErrorCode::tool_execution,
                .message = "Tool has no execute handler: " + definition_.name,
            });
        }
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
};

namespace detail {

template <typename T>
struct tool_return_traits {
    using value_type = T;
    static constexpr bool is_result = false;
};

template <typename T>
struct tool_return_traits<std::expected<T, Error>> {
    using value_type = T;
    static constexpr bool is_result = true;
};

template <typename Handler, typename Input>
decltype(auto) invoke_tool(Handler& handler, const Input& input, const ToolContext& context)
{
    if constexpr (std::invocable<Handler&, const Input&, const ToolContext&>) {
        return std::invoke(handler, input, context);
    } else {
        return std::invoke(handler, input);
    }
}

template <typename Input, typename Output, typename Handler>
[[nodiscard]] Tool::Executor make_tool_executor(Handler&& handler)
{
    using HandlerType = std::decay_t<Handler>;
    using HandlerReturn = std::remove_cvref_t<decltype(invoke_tool(
        std::declval<HandlerType&>(), std::declval<const Input&>(), std::declval<const ToolContext&>()))>;
    using ReturnTraits = tool_return_traits<HandlerReturn>;
    static_assert(std::same_as<typename ReturnTraits::value_type, Output>,
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

template <typename Input, typename Output>
class ExecutableTool : public Tool {
    public:
    template <typename Handler>
    ExecutableTool(std::string name, std::string description, Handler&& handler)
        : Tool(
              make_tool<Input>(std::move(name), std::move(description)),
              detail::make_tool_executor<Input, Output>(std::forward<Handler>(handler))) {}

    [[nodiscard]] Result<Output> decode_output(const ToolResult& result) const
    {
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
[[nodiscard]] ExecutableTool<Input, Output> tool(
    std::string name, std::string description, Handler&& handler)
{
    return ExecutableTool<Input, Output>{
        std::move(name), std::move(description), std::forward<Handler>(handler)};
}

template <typename Client>
[[nodiscard]] Result<GenerationResponse> run_tool_loop(
    const Client& client,
    GenerationRequest request,
    const std::vector<Tool>& tools,
    ToolLoopOptions options = {})
{
    request.tools.clear();
    request.tools.reserve(tools.size());
    for (const auto& tool : tools) {
        request.tools.push_back(tool.definition());
    }

    auto response = client.generate(request);
    if (!response) {
        return std::unexpected(response.error());
    }

    std::vector<ToolResult> tool_results;
    std::size_t round = 0;
    while (!response->tool_calls.empty()) {
        if (round >= options.max_rounds) {
            return std::unexpected(Error{
                .code = ErrorCode::tool_loop_limit,
                .message = "The model exceeded the configured tool-call round limit.",
            });
        }
        if (!response->continuation_token) {
            return std::unexpected(Error{
                .code = ErrorCode::provider_response,
                .message = "The provider did not return continuation state for its tool calls.",
            });
        }

        GenerationRequest follow_up{
            .tools = request.tools,
            .structured_output = request.structured_output,
            .continuation_token = response->continuation_token,
        };
        for (const auto& call : response->tool_calls) {
            const auto tool = std::ranges::find_if(tools, [&call](const auto& candidate) {
                return candidate.definition().name == call.name;
            });
            if (tool == tools.end()) {
                return std::unexpected(Error{
                    .code = ErrorCode::tool_not_found,
                    .message = "The model requested an unregistered tool: " + call.name,
                });
            }

            auto output = tool->execute(call, ToolContext{.call_id = call.id, .round = round});
            if (!output) {
                return std::unexpected(output.error());
            }
            tool_results.push_back(ToolResult{
                .call_id = call.id,
                .name = call.name,
                .output = *output,
            });
            follow_up.messages.push_back(Message{
                .role = MessageRole::tool,
                .content = std::move(*output),
                .tool_call_id = call.id,
            });
        }

        ++round;
        response = client.generate(follow_up);
        if (!response) {
            return std::unexpected(response.error());
        }
    }

    response->tool_results = std::move(tool_results);
    return response;
}

} // namespace cail
