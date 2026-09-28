#pragma once

#include <cail/language_model.hpp>
#include <cail/tool.hpp>

#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cail {

struct AgentConfig {
  LanguageModel model;
  std::string instructions;
  std::vector<Tool> tools;
};

class Agent {
public:
  explicit Agent(AgentConfig config)
      : model_(std::move(config.model)), instructions_(std::move(config.instructions)),
        tools_(std::move(config.tools)) {}

  [[nodiscard]] Result<GenerationResponse> generate(std::string_view prompt,
                                                    ToolLoopOptions options = {}) const {
    return generate(request(prompt), std::move(options));
  }

  [[nodiscard]] Result<GenerationResponse> generate(GenerationRequest request,
                                                    ToolLoopOptions options = {}) const {
    return run_tool_loop(model_, prepare(std::move(request)), tools_, std::move(options));
  }

  [[nodiscard]] Result<GenerationResponse> stream(std::string_view prompt,
                                                  const StreamHandler& on_event,
                                                  ToolLoopOptions options = {}) const {
    return stream(request(prompt), on_event, std::move(options));
  }

  [[nodiscard]] Result<GenerationResponse> stream(GenerationRequest request,
                                                  const StreamHandler& on_event,
                                                  ToolLoopOptions options = {}) const {
    return stream_tool_loop(model_, prepare(std::move(request)), tools_, on_event,
                            std::move(options));
  }

private:
  [[nodiscard]] static GenerationRequest request(std::string_view prompt) {
    return GenerationRequest{
        .messages = {Message{.content = {TextPart{.text = std::string{prompt}}}}},
    };
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
};

} // namespace cail
