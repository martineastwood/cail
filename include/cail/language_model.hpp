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
  bool async_generation{};
  bool image_input{};
  bool tools{};
  bool structured_output{};
  bool reasoning{};
  bool continuation{};
};

class LanguageModel {
public:
  using GenerationCompletion = std::function<void(Result<GenerationResponse>)>;
  using GenerateFunction =
      std::function<Result<GenerationResponse>(const GenerationRequest&, std::stop_token)>;
  using GenerateAsyncFunction =
      std::function<Result<void>(GenerationRequest, GenerationCompletion, std::stop_token)>;
  using StreamFunction = std::function<Result<GenerationResponse>(
      const GenerationRequest&, const StreamHandler&, std::stop_token)>;

  LanguageModel() = default;
  explicit LanguageModel(GenerateFunction generate) : generate_(std::move(generate)) {}
  LanguageModel(GenerateFunction generate, StreamFunction stream,
                AdapterCapabilities adapter_capabilities = {},
                GenerateAsyncFunction generate_async = {})
      : generate_(std::move(generate)), stream_(std::move(stream)),
        adapter_capabilities_(std::move(adapter_capabilities)),
        generate_async_(std::move(generate_async)) {
    adapter_capabilities_.streaming = static_cast<bool>(stream_);
    adapter_capabilities_.async_generation = static_cast<bool>(generate_async_);
  }

  [[nodiscard]] explicit operator bool() const noexcept { return static_cast<bool>(generate_); }
  [[nodiscard]] const AdapterCapabilities& adapter_capabilities() const noexcept {
    return adapter_capabilities_;
  }

  [[nodiscard]] Result<GenerationResponse> generate(const GenerationRequest& request,
                                                    std::stop_token stop = {}) const {
    if (!generate_) {
      return std::unexpected(Error{
          .code = ErrorCode::invalid_configuration,
          .message = "The language model has no provider implementation.",
      });
    }
    if (stop.stop_requested()) {
      return std::unexpected(generation_cancelled_error());
    }
    return generate_(request, stop);
  }

  [[nodiscard]] Result<void> generate_async(GenerationRequest request,
                                            GenerationCompletion complete,
                                            std::stop_token stop = {}) const {
    if (!generate_async_ || !complete) {
      return std::unexpected(
          Error{.code = ErrorCode::invalid_configuration,
                .message = "Async generation requires a capable model and completion handler."});
    }
    if (stop.stop_requested()) {
      return std::unexpected(generation_cancelled_error());
    }
    return generate_async_(std::move(request), std::move(complete), stop);
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
  GenerateAsyncFunction generate_async_;
};

} // namespace cail
