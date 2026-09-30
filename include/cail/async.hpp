#pragma once

#include <cail/detail/async.hpp>
#include <cail/generation.hpp>

#include <deque>

namespace cail {

// The scheduler must accept each task exactly once, or throw before accepting it.
// Tasks may run inline. Stream callbacks are serialized even on concurrent executors.
struct AsyncOptions {
  std::function<void(std::function<void()>)> schedule;
  std::size_t max_pending_events{256};
};

namespace detail {

inline Error async_callback_error(std::string message) {
  return {.code = ErrorCode::invalid_configuration, .message = std::move(message)};
}

template <typename T, typename Completion>
auto scheduled_completion(Completion complete, const AsyncOptions& options) {
  auto done = complete_once<T>(std::move(complete));
  return [done, schedule = options.schedule](Result<T> result) mutable {
    if (!schedule) {
      done(std::move(result));
      return;
    }
    auto owned = std::make_shared<Result<T>>(std::move(result));
    try {
      schedule([done, owned]() mutable { done(std::move(*owned)); });
    } catch (...) {
      done(std::unexpected(async_callback_error("The callback scheduler rejected completion.")));
    }
  };
}

class StreamDelivery : public std::enable_shared_from_this<StreamDelivery> {
public:
  StreamDelivery(StreamHandler handler, std::function<void(Result<GenerationResponse>)> complete,
                 AsyncOptions options, std::stop_token stop)
      : handler_(std::move(handler)),
        complete_(complete_once<GenerationResponse>(std::move(complete))),
        options_(std::move(options)), application_(stop, [this] { stop_.request_stop(); }) {}

  std::stop_token token() const { return stop_.get_token(); }

  void event(const StreamEvent& event) {
    bool launch = false;
    bool overflow = false;
    {
      std::lock_guard lock(mutex_);
      if (terminal_ || error_ || token().stop_requested())
        return;
      if (events_.size() >= options_.max_pending_events) {
        error_ = Error{.code = ErrorCode::backpressure,
                       .message = "The async stream consumer exceeded max_pending_events."};
        events_.clear();
        overflow = true;
      } else {
        events_.push_back(event);
        launch = !std::exchange(scheduled_, true);
      }
    }
    if (overflow)
      stop_.request_stop();
    if (launch)
      schedule();
  }

  void finish(Result<GenerationResponse> result) {
    bool launch;
    {
      std::lock_guard lock(mutex_);
      if (terminal_)
        return;
      terminal_ = true;
      result_ = std::move(result);
      launch = !std::exchange(scheduled_, true);
    }
    if (launch)
      schedule();
  }

  void abort() {
    {
      std::lock_guard lock(mutex_);
      terminal_ = true;
      events_.clear();
      result_.reset();
    }
    stop_.request_stop();
  }

private:
  void schedule() {
    auto self = shared_from_this();
    try {
      if (options_.schedule)
        options_.schedule([self] { self->drain(); });
      else
        drain();
    } catch (...) {
      {
        std::lock_guard lock(mutex_);
        error_ = async_callback_error("The callback scheduler rejected stream delivery.");
        events_.clear();
      }
      stop_.request_stop();
      // A rejected scheduler has no thread on which to report the failure.
      drain();
    }
  }

  void drain() {
    while (true) {
      std::optional<StreamEvent> event;
      std::optional<Result<GenerationResponse>> result;
      {
        std::lock_guard lock(mutex_);
        if (error_ || token().stop_requested())
          events_.clear();
        if (!events_.empty()) {
          event = std::move(events_.front());
          events_.pop_front();
        } else if (result_) {
          result = std::move(result_);
          result_.reset();
          if (error_)
            *result = std::unexpected(*error_);
          else if (token().stop_requested())
            *result = std::unexpected(generation_cancelled_error());
        } else {
          scheduled_ = false;
          return;
        }
      }
      if (result) {
        complete_(std::move(*result));
        return;
      }
      try {
        if (!token().stop_requested())
          handler_(*event);
      } catch (...) {
        {
          std::lock_guard lock(mutex_);
          error_ = async_callback_error("The async stream event handler threw an exception.");
        }
        stop_.request_stop();
      }
    }
  }

  StreamHandler handler_;
  std::function<void(Result<GenerationResponse>)> complete_;
  AsyncOptions options_;
  std::stop_source stop_;
  std::stop_callback<std::function<void()>> application_;
  std::mutex mutex_;
  std::deque<StreamEvent> events_;
  std::optional<Result<GenerationResponse>> result_;
  std::optional<Error> error_;
  bool scheduled_{};
  bool terminal_{};
};

} // namespace detail
} // namespace cail
