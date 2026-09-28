#pragma once

#include <cail/error.hpp>
#include <cail/generation.hpp>

#include <functional>
#include <stop_token>
#include <utility>

namespace cail {

// Adapter wire capabilities. Does not guarantee the selected model accepts every feature.
// `streaming` is derived from whether a stream function was provided.
struct AdapterCapabilities {
  bool streaming{};
  bool image_input{};
  bool tools{};
  bool structured_output{};
  bool reasoning{};
  bool continuation{};
};

class LanguageModel {
public:
  using GenerateFunction = std::function<Result<GenerationResponse>(const GenerationRequest&)>;
  using StreamFunction = std::function<Result<GenerationResponse>(
      const GenerationRequest&, const StreamHandler&, std::stop_token)>;

  LanguageModel() = default;
  explicit LanguageModel(GenerateFunction generate) : generate_(std::move(generate)) {}
  LanguageModel(GenerateFunction generate, StreamFunction stream,
                AdapterCapabilities adapter_capabilities = {})
      : generate_(std::move(generate)), stream_(std::move(stream)),
        adapter_capabilities_(std::move(adapter_capabilities)) {
    adapter_capabilities_.streaming = static_cast<bool>(stream_);
  }

  [[nodiscard]] explicit operator bool() const noexcept { return static_cast<bool>(generate_); }
  [[nodiscard]] const AdapterCapabilities& adapter_capabilities() const noexcept {
    return adapter_capabilities_;
  }

  [[nodiscard]] Result<GenerationResponse> generate(const GenerationRequest& request) const {
    if (!generate_) {
      return std::unexpected(Error{
          .code = ErrorCode::invalid_configuration,
          .message = "The language model has no provider implementation.",
      });
    }
    return generate_(request);
  }

  [[nodiscard]] Result<GenerationResponse> stream(const GenerationRequest& request,
                                                  const StreamHandler& on_event,
                                                  std::stop_token stop = {}) const {
    if (!stream_) {
      return std::unexpected(Error{
          .code = ErrorCode::invalid_configuration,
          .message = "The language model provider does not support streaming.",
      });
    }
    if (!on_event) {
      return std::unexpected(Error{
          .code = ErrorCode::invalid_configuration,
          .message = "Streaming requires an event handler.",
      });
    }
    return stream_(request, on_event, stop);
  }

  [[nodiscard]] Result<GenerationResponse>
  stream(std::string_view prompt, const StreamHandler& on_event, std::stop_token stop = {}) const {
    return stream(detail::user_prompt_request(prompt), on_event, stop);
  }

private:
  GenerateFunction generate_;
  StreamFunction stream_;
  AdapterCapabilities adapter_capabilities_;
};

} // namespace cail
