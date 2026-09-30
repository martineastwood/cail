#pragma once

#include <cail/error.hpp>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stop_token>
#include <thread>
#include <utility>
#include <vector>

namespace cail::detail {

// A terminal callback has no error channel of its own. Contain its exceptions
// and release its captures after completion or rejected initiation.
template <typename T> class AsyncCompletion {
public:
  explicit AsyncCompletion(std::function<void(Result<T>)> complete)
      : state_(std::make_shared<State>()) {
    state_->complete = std::move(complete);
  }

  void operator()(Result<T> result) const noexcept {
    auto callback = take();
    try {
      if (callback)
        callback(std::move(result));
    } catch (...) {
      // Completion has already been delivered; never retry it.
    }
  }

  [[nodiscard]] bool reject() const { return static_cast<bool>(take()); }

private:
  std::function<void(Result<T>)> take() const {
    std::lock_guard lock(state_->mutex);
    return std::exchange(state_->complete, {});
  }
  struct State {
    std::mutex mutex;
    std::function<void(Result<T>)> complete;
  };
  std::shared_ptr<State> state_;
};

template <typename T, typename Completion> auto complete_once(Completion complete) {
  return AsyncCompletion<T>{std::function<void(Result<T>)>{std::move(complete)}};
}

// Rejected initiation releases the callback even if a custom adapter retained it.
template <typename T, typename Completion, typename Start>
Result<void> initiate_async(Completion complete, Start start) {
  auto done = complete_once<T>(std::move(complete));
  try {
    auto result = start(done);
    if (!result && !done.reject())
      return {};
    return result;
  } catch (const std::exception& error) {
    if (!done.reject())
      return {};
    return std::unexpected(
        Error{.code = ErrorCode::invalid_configuration,
              .message = std::string{"Async initiation failed: "} + error.what()});
  } catch (...) {
    if (!done.reject())
      return {};
    return std::unexpected(
        Error{.code = ErrorCode::invalid_configuration, .message = "Async initiation failed."});
  }
}

// Blocking adapters use a fixed number of joined workers. Admission never waits,
// and shutdown cancels queued work and cooperatively stops active work.
class AsyncWorkers {
public:
  using Task = std::function<void(std::stop_token)>;

  explicit AsyncWorkers(std::size_t count = 4, std::size_t capacity = 256) : capacity_(capacity) {
    for (std::size_t i = 0; i < count; ++i)
      workers_.emplace_back([this](std::stop_token stop) { run(stop); });
  }

  ~AsyncWorkers() { shutdown(); }

  void shutdown() {
    {
      std::lock_guard lock(mutex_);
      if (closing_)
        return;
      closing_ = true;
    }
    for (auto& worker : workers_)
      worker.request_stop();
    ready_.notify_all();
    // Join before destroying the queue and synchronization objects.
    for (auto& worker : workers_)
      worker.join();
  }

  [[nodiscard]] Result<void> post(Task task, std::stop_token stop = {}) {
    auto job = std::make_shared<Job>();
    job->run = std::move(task);
    std::vector<std::shared_ptr<Job>> retired;
    {
      std::lock_guard lock(mutex_);
      for (auto i = tasks_.begin(); i != tasks_.end();) {
        if ((*i)->claimed.load()) {
          retired.push_back(std::move(*i));
          i = tasks_.erase(i);
        } else {
          ++i;
        }
      }
      if (closing_ || tasks_.size() >= capacity_)
        return std::unexpected(
            Error{.code = ErrorCode::invalid_configuration,
                  .message = "The async worker queue is full or shutting down."});
      tasks_.push_back(job);
    }
    job->cancel.emplace(stop, [weak = std::weak_ptr(job), stop] {
      if (auto active = weak.lock(); active && !active->claimed.exchange(true))
        active->run(stop);
    });
    ready_.notify_one();
    return {};
  }

private:
  struct Job {
    Task run;
    std::atomic<bool> claimed{false};
    std::optional<std::stop_callback<std::function<void()>>> cancel;
  };

  void run(std::stop_token stop) {
    while (true) {
      std::shared_ptr<Job> job;
      {
        std::unique_lock lock(mutex_);
        ready_.wait(lock, stop, [&] { return closing_ || !tasks_.empty(); });
        if (tasks_.empty())
          return;
        job = std::move(tasks_.front());
        tasks_.pop_front();
      }
      if (!job->claimed.exchange(true))
        job->run(stop);
    }
  }

  std::size_t capacity_;
  std::mutex mutex_;
  std::condition_variable_any ready_;
  std::deque<std::shared_ptr<Job>> tasks_;
  bool closing_{};
  std::vector<std::jthread> workers_;
};

// I/O shutdown stops any still-live adapters and retry timers before stopping I/O.
// Weak registration avoids imposing an initialization order on custom models.
struct AsyncWorkerRegistry {
  std::mutex mutex;
  std::vector<std::function<void()>> shutdown_tasks;

  template <typename Runtime> void add(const std::shared_ptr<Runtime>& runtime) {
    std::lock_guard lock(mutex);
    shutdown_tasks.push_back([weak = std::weak_ptr(runtime)] {
      if (auto active = weak.lock())
        active->shutdown();
    });
  }

  void shutdown() {
    std::vector<std::function<void()>> tasks;
    {
      std::lock_guard lock(mutex);
      tasks = shutdown_tasks;
    }
    for (auto& task : tasks)
      task();
  }
};

inline AsyncWorkerRegistry& async_worker_registry() {
  static AsyncWorkerRegistry registry;
  return registry;
}

inline AsyncWorkers& tool_workers() {
  static auto workers = [] {
    auto pool = std::make_shared<AsyncWorkers>();
    async_worker_registry().add(pool);
    return pool;
  }();
  return *workers;
}

inline AsyncWorkers& memory_workers() {
  static auto workers = [] {
    auto pool = std::make_shared<AsyncWorkers>();
    async_worker_registry().add(pool);
    return pool;
  }();
  return *workers;
}

// Link application cancellation and worker shutdown without changing the API.
struct LinkedStop {
  std::stop_source source;
  std::stop_callback<std::function<void()>> application;
  std::stop_callback<std::function<void()>> worker;

  LinkedStop(std::stop_token application_stop, std::stop_token worker_stop)
      : application(application_stop, [this] { source.request_stop(); }),
        worker(worker_stop, [this] { source.request_stop(); }) {}
};

} // namespace cail::detail
