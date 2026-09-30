#pragma once

#include <cail/error.hpp>
#include <cail/generation.hpp>
#include <cail/task.hpp>

#include <exception>
#include <functional>
#include <stop_token>
#include <thread>
#include <utility>

namespace cail {

// Adapter wire capabilities. Does not guarantee the selected model accepts every feature.
// Streaming and async generation are derived from the supplied functions.
struct AdapterCapabilities {
  bool streaming{};
  bool async_generation{};
  bool async_streaming{};
  bool image_input{};
  bool pdf_input{};
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
  // Return an initiation error without calling completion, or complete exactly once.
  // Completion may run inline or on the completing thread. Synchronize shared state,
  // return promptly, and do not block on other async completions in a callback.
  // The operation must own everything needed until it finishes. Exceptions from
  // terminal callbacks are contained; they do not cause another completion.
  using GenerateAsyncFunction =
      std::function<Result<void>(GenerationRequest, GenerationCompletion, std::stop_token)>;
  using StreamFunction = std::function<Result<GenerationResponse>(
      const GenerationRequest&, const StreamHandler&, std::stop_token)>;

  using StreamAsyncFunction = std::function<Result<void>(GenerationRequest, StreamHandler,
                                                         GenerationCompletion, std::stop_token)>;

  LanguageModel() = default;
  explicit LanguageModel(GenerateFunction generate) : generate_(std::move(generate)) {}
  LanguageModel(GenerateFunction generate, StreamFunction stream,
                AdapterCapabilities adapter_capabilities = {},
                GenerateAsyncFunction generate_async = {}, StreamAsyncFunction stream_async = {})
      : generate_(std::move(generate)), stream_(std::move(stream)),
        adapter_capabilities_(std::move(adapter_capabilities)),
        generate_async_(std::move(generate_async)), stream_async_(std::move(stream_async)) {
    adapter_capabilities_.streaming = static_cast<bool>(stream_);
    adapter_capabilities_.async_streaming = static_cast<bool>(stream_async_);
    adapter_capabilities_.async_generation = static_cast<bool>(generate_async_);
  }

  [[nodiscard]] explicit operator bool() const noexcept {
    return generate_ || stream_ || generate_async_ || stream_async_;
  }
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
                                            std::stop_token stop = {},
                                            AsyncOptions options = {}) const {
    if (!generate_async_ || !complete) {
      return std::unexpected(
          Error{.code = ErrorCode::invalid_configuration,
                .message = "Async generation requires a capable model and completion handler."});
    }
    if (stop.stop_requested()) {
      return std::unexpected(generation_cancelled_error());
    }
    return detail::initiate_async<GenerationResponse>(
        detail::scheduled_completion<GenerationResponse>(std::move(complete), options),
        [&](auto done) { return generate_async_(std::move(request), std::move(done), stop); });
  }

  [[nodiscard]] Task<Result<GenerationResponse>> generate_async(GenerationRequest request,
                                                                std::stop_token stop = {},
                                                                AsyncOptions options = {}) const {
    return detail::await_result<GenerationResponse>(
        [model = *this, request = std::move(request),
         options = std::move(options)](auto complete, auto token, auto) mutable {
          return model.generate_async(std::move(request), std::move(complete), token,
                                      std::move(options));
        },
        stop);
  }

  // Events run serially. Inline handlers gate further reads; scheduled handlers
  // use a bounded queue and completion follows all delivered events.
  [[nodiscard]] Result<void> stream_async(GenerationRequest request, StreamHandler on_event,
                                          GenerationCompletion complete, std::stop_token stop = {},
                                          AsyncOptions options = {}) const {
    if (!stream_async_ || !on_event || !complete)
      return std::unexpected(Error{
          .code = ErrorCode::invalid_configuration,
          .message = "Async streaming requires a capable model, event and completion handlers."});
    if (stop.stop_requested())
      return std::unexpected(generation_cancelled_error());
    if (!options.max_pending_events)
      return std::unexpected(detail::async_callback_error("max_pending_events must be positive."));
    auto delivery = std::make_shared<detail::StreamDelivery>(
        std::move(on_event), std::move(complete), std::move(options), stop);
    auto started = detail::initiate_async<GenerationResponse>(
        [delivery](Result<GenerationResponse> result) { delivery->finish(std::move(result)); },
        [&](auto done) {
          return stream_async_(
              std::move(request), [delivery](const StreamEvent& event) { delivery->event(event); },
              std::move(done), delivery->token());
        });
    if (!started)
      delivery->abort();
    return started;
  }

  [[nodiscard]] Task<Result<GenerationResponse>> stream_async(GenerationRequest request,
                                                              StreamHandler on_event,
                                                              std::stop_token stop = {},
                                                              AsyncOptions options = {}) const {
    return detail::await_result<GenerationResponse>(
        [model = *this, request = std::move(request), on_event = std::move(on_event),
         options = std::move(options)](auto complete, auto token, auto callbacks) mutable {
          if (!options.schedule)
            options.schedule = std::move(callbacks.schedule);
          return model.stream_async(std::move(request), std::move(on_event), std::move(complete),
                                    token, std::move(options));
        },
        stop);
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
    if (stop.stop_requested())
      return std::unexpected(generation_cancelled_error());
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
  StreamAsyncFunction stream_async_;
};

} // namespace cail
