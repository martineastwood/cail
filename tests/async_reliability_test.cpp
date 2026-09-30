#include "test_support.hpp"

#include <cail/cail.hpp>

#include <atomic>
#include <chrono>
#include <future>
#include <latch>
#include <stdexcept>
#include <thread>
#include <utility>

namespace test {

using namespace std::chrono_literals;

cail::LanguageModel tool_model() {
  return cail::LanguageModel(
      {}, {}, {},
      [](const cail::GenerationRequest& request,
         const cail::LanguageModel::GenerationCompletion& complete,
         const std::stop_token&) -> cail::Result<void> {
        if (request.step == 0) {
          complete(cail::GenerationResponse{
              .tool_calls = {{.id = "call", .name = "count", .arguments = R"({"query":"abc"})"}}});
        } else {
          complete(cail::GenerationResponse{.text = "done"});
        }
        return {};
      });
}

void test_delayed_tools() {
  std::vector<std::function<void(cail::Result<ToolOutput>)>> pending;
  auto count = cail::async_tool<ToolInput, ToolOutput>(
      "count", "Count", [&](const ToolInput& input, const cail::ToolContext&, auto complete) {
        check(input.query == "abc", "async tool owns its decoded input");
        pending.push_back(std::move(complete));
      });
  int completed = 0;
  for (int i = 0; i < 64; ++i) {
    check(cail::generate_text_async({.model = tool_model(), .prompt = "count", .tools = {count}},
                                    [&](auto result) {
                                      check(
                                          result && result->tool_results.size() == 1 &&
                                              result->tool_results[0].output == R"({"count":3})",
                                          "a delayed async tool resumes and preserves its result");
                                      ++completed;
                                    })
              .has_value(),
          "delayed tool generation starts");
  }
  check(pending.size() == 64 && completed == 0,
        "pending async tools do not consume the bounded blocking workers");
  for (auto& complete : pending) {
    complete(ToolOutput{.count = 3});
    complete(ToolOutput{.count = 999});
  }
  check(completed == 64, "duplicate tool completions do not restart or finish the loop twice");
}

void test_tool_cancellation_and_errors() {
  std::function<void(cail::Result<ToolOutput>)> pending;
  std::stop_source stop;
  int callbacks = 0;
  auto count = cail::async_tool<ToolInput, ToolOutput>(
      "count", "Count", [&](const ToolInput&, const cail::ToolContext& context, auto complete) {
        check(context.stop == stop.get_token(), "async tools receive application cancellation");
        pending = std::move(complete);
      });
  check(cail::generate_text_async({.model = tool_model(),
                                   .prompt = "count",
                                   .tools = {count},
                                   .tool_loop = {.stop = stop.get_token()}},
                                  [&](auto result) {
                                    ++callbacks;
                                    check(!result &&
                                              result.error().code == cail::ErrorCode::cancelled,
                                          "cancelling a pending async tool completes generation");
                                  })
            .has_value(),
        "cancellable async tool starts");
  stop.request_stop();
  check(callbacks == 1, "tool cancellation does not wait for the tool callback");
  pending(ToolOutput{.count = 3});
  check(callbacks == 1, "late tool completion after cancellation is ignored");

  auto throwing = cail::async_tool<ToolInput, ToolOutput>(
      "count", "Count", [](const ToolInput&, const cail::ToolContext&, const auto&) {
        throw std::runtime_error("tool failed");
      });
  callbacks = 0;
  check(cail::generate_text_async(
            {.model = tool_model(), .prompt = "count", .tools = {throwing}},
            [&](auto result) {
              ++callbacks;
              check(!result && result.error().code == cail::ErrorCode::tool_execution,
                    "async tool initiation exceptions reach completion");
            }).has_value() &&
            callbacks == 1,
        "throwing async tools finish once");

  std::promise<void> entered;
  auto active = entered.get_future();
  std::promise<void> release;
  auto released = release.get_future().share();
  auto blocking = cail::tool<ToolInput, ToolOutput>(
      "count", "Count", [&](const ToolInput&, const cail::ToolContext& context) {
        entered.set_value();
        released.wait();
        check(context.stop.stop_requested(), "application stop reaches an active synchronous tool");
        return ToolOutput{};
      });
  std::promise<cail::Result<cail::GenerationResponse>> completion;
  auto result = completion.get_future();
  std::stop_source cancel;
  check(cail::generate_text_async({.model = tool_model(),
                                   .prompt = "count",
                                   .tools = {blocking},
                                   .tool_loop = {.stop = cancel.get_token()}},
                                  [&](auto value) { completion.set_value(std::move(value)); })
            .has_value(),
        "blocking tool runs away from the initiating thread");
  active.get();
  cancel.request_stop();
  check(result.wait_for(10ms) == std::future_status::timeout,
        "active synchronous tools finish before cancellation completes");
  release.set_value();
  auto cancelled = result.get();
  check(!cancelled && cancelled.error().code == cail::ErrorCode::cancelled,
        "a cancelled synchronous tool does not start another model step");
}

void test_tool_completion_cancellation_race() {
  for (int i = 0; i < 64; ++i) {
    std::function<void(cail::Result<ToolOutput>)> pending;
    auto count = cail::async_tool<ToolInput, ToolOutput>(
        "count", "Count", [&](const ToolInput&, const cail::ToolContext&, auto complete) {
          pending = std::move(complete);
        });
    std::stop_source stop;
    std::atomic<int> callbacks{};
    check(cail::generate_text_async(
              {.model = tool_model(),
               .prompt = "count",
               .tools = {count},
               .tool_loop = {.stop = stop.get_token()}},
              [&](auto response) {
                ++callbacks;
                check(response || response.error().code == cail::ErrorCode::cancelled,
                      "racing tool completion and cancellation produce a valid outcome");
              })
              .has_value(),
          "racing async tool starts");
    std::latch ready(2);
    std::jthread cancel([&] {
      ready.arrive_and_wait();
      stop.request_stop();
    });
    std::jthread reply([&] {
      ready.arrive_and_wait();
      pending(ToolOutput{.count = 3});
    });
    cancel.join();
    reply.join();
    check(callbacks == 1, "racing tool completion and cancellation finish exactly once");
  }
}

void test_callbacks_and_initiation() {
  int callbacks = 0;
  cail::LanguageModel model(
      {}, {}, {}, [](const auto&, const auto& complete, const auto&) -> cail::Result<void> {
        complete(cail::GenerationResponse{.text = "done"});
        complete(cail::GenerationResponse{});
        throw std::runtime_error("after completion");
      });
  check(model.generate_async({},
                             [&](auto result) {
                               ++callbacks;
                               check(result && result->text == "done",
                                     "first model completion wins");
                               throw std::runtime_error("completion failed");
                             })
                .has_value() &&
            callbacks == 1,
        "terminal callback exceptions are contained and do not repeat completion");

  cail::LanguageModel::GenerationCompletion retained;
  cail::LanguageModel rejecting(
      {}, {}, {}, [&](const auto&, auto complete, const auto&) -> cail::Result<void> {
        retained = std::move(complete);
        return std::unexpected(cail::Error{.code = cail::ErrorCode::transport});
      });
  callbacks = 0;
  check(!rejecting.generate_async({}, [&](auto) { ++callbacks; }),
        "initiation errors are returned directly");
  retained(cail::GenerationResponse{});
  check(callbacks == 0, "rejected initiation invalidates a retained callback");

  cail::LanguageModel throwing({}, {}, {},
                               [](const auto&, const auto&, const auto&) -> cail::Result<void> {
                                 throw std::runtime_error("start failed");
                               });
  check(!throwing.generate_async({}, [&](auto) { ++callbacks; }) && callbacks == 0,
        "model initiation exceptions do not escape or invoke completion");

  cail::EmbeddingModel embeddings(
      {}, [](const auto&, const auto& complete, const auto&) -> cail::Result<void> {
        complete(cail::EmbeddingBatch{.embeddings = {cail::Embedding{}}});
        complete(cail::EmbeddingBatch{});
        return {};
      });
  check(embeddings
                .embed_async("hello",
                             [&](auto result) {
                               ++callbacks;
                               check(result.has_value(), "single embedding completes successfully");
                               throw std::runtime_error("embedding callback failed");
                             })
                .has_value() &&
            callbacks == 1,
        "embedding completions are contained and delivered once");
}

void test_bounded_workers_and_shutdown() {
  auto workers = std::make_unique<cail::detail::AsyncWorkers>(1, 1);
  std::promise<void> entered;
  auto active = entered.get_future();
  std::promise<void> stopped;
  auto finished = stopped.get_future();
  check(workers
            ->post([&](std::stop_token stop) {
              std::mutex mutex;
              std::unique_lock lock(mutex);
              std::condition_variable_any ready;
              entered.set_value();
              ready.wait(lock, std::move(stop), [] { return false; });
              stopped.set_value();
            })
            .has_value(),
        "first worker task starts");
  active.get();
  bool queued_cancelled = false;
  check(
      workers->post([&](const std::stop_token& stop) { queued_cancelled = stop.stop_requested(); })
          .has_value(),
      "one task can wait in the bounded queue");
  check(!workers->post([](const auto&) {}), "worker overload is rejected without blocking");
  workers.reset();
  check(finished.wait_for(0ms) == std::future_status::ready && queued_cancelled,
        "worker destruction cancels, drains, and joins active and queued tasks");
}

void test_streaming() {
  std::vector<cail::StreamHandler> handlers;
  std::vector<cail::LanguageModel::GenerationCompletion> pending;
  std::vector<std::stop_token> tokens;
  cail::LanguageModel model(
      {}, {}, {}, {}, [&](auto, auto handler, auto complete, auto stop) -> cail::Result<void> {
        handlers.push_back(std::move(handler));
        pending.push_back(std::move(complete));
        tokens.push_back(stop);
        return {};
      });
  check(model.adapter_capabilities().async_streaming && !model.adapter_capabilities().streaming,
        "native async streaming capability is independent of synchronous streaming");
  int callbacks = 0, events = 0;
  std::vector<std::stop_source> stops(12);
  for (auto& stop : stops)
    check(model
              .stream_async(
                  {}, [&](const auto&) { ++events; },
                  [&](auto result) {
                    check(!result, "cancelled native stream returns an error");
                    ++callbacks;
                  },
                  stop.get_token())
              .has_value(),
          "native streams start without occupying blocking workers");
  check(pending.size() == 12, "more than four native streams start immediately");
  for (std::size_t i = 0; i < stops.size(); ++i) {
    stops[i].request_stop();
    check(tokens[i].stop_requested(), "native streams receive linked cancellation");
    handlers[i](cail::TextDelta{.text = "late"});
    pending[i](std::unexpected(cail::generation_cancelled_error()));
    pending[i](cail::GenerationResponse{});
  }
  check(callbacks == 12 && events == 0, "cancelled events are suppressed and completion runs once");
  cail::LanguageModel emitting({}, {}, {}, {},
                               [](auto, auto handler, auto complete, auto) -> cail::Result<void> {
                                 handler(cail::TextDelta{.text = "hello"});
                                 complete(cail::GenerationResponse{});
                                 return {};
                               });
  bool failed = false;
  check(emitting.stream_async(
                    {}, [](const auto&) { throw std::runtime_error("event failed"); },
                    [&](auto result) { failed = !result; })
                .has_value() &&
            failed,
        "event handler exceptions complete native streaming with an error");
  check(emitting
            .stream_async(
                {}, [](const auto&) {}, [](auto) { throw std::runtime_error("terminal failed"); })
            .has_value(),
        "terminal stream exceptions are contained");
  cail::LanguageModel sync_only(
      {}, [](const auto&, const auto&, const auto&) { return cail::GenerationResponse{}; });
  check(!sync_only.adapter_capabilities().async_streaming &&
            !sync_only.stream_async(
                {}, [](const auto&) {}, [](auto) {}),
        "synchronous adapters do not advertise native async streaming");
}

void test_conversation_overlap() {
  std::vector<cail::LanguageModel::GenerationCompletion> pending;
  auto model = cail::LanguageModel(
      {}, {}, {}, [&](const auto&, auto complete, const auto&) -> cail::Result<void> {
        pending.push_back(std::move(complete));
        return {};
      });
  auto memory = std::make_shared<InlineMemory>();
  cail::Agent first({.model = model, .memory = memory, .conversation_id = "same"});
  cail::Agent second({.model = model, .memory = memory, .conversation_id = "same"});
  int callbacks = 0;
  check(first
            .generate_async("first",
                            [&](auto result) {
                              check(result.has_value(), "first conversation turn succeeds");
                              ++callbacks;
                            })
            .has_value(),
        "first conversation turn starts");
  const auto rejected = second.generate_async("overlap", [&](auto) { ++callbacks; });
  check(!rejected && rejected.error().code == cail::ErrorCode::memory && pending.size() == 1,
        "overlap across agents sharing a conversation is rejected before generation");
  check(second.generate_async("other", [&](auto) { ++callbacks; }, {.conversation_id = "other"})
            .has_value(),
        "different conversations can run concurrently");
  pending[0](cail::GenerationResponse{.text = "done"});
  pending[1](cail::GenerationResponse{.text = "done"});
  check(callbacks == 2 && memory->load("same")->size() == 2 && memory->load("other")->size() == 2,
        "concurrent conversations store only their own complete turns");
  check(first.generate_async("next", [&](auto) { ++callbacks; }).has_value(),
        "completion releases the conversation for its next turn");
  pending[2](std::unexpected(cail::Error{.code = cail::ErrorCode::transport}));
  check(first.generate_async("after failure", [&](auto) { ++callbacks; }).has_value(),
        "failed completion also releases the conversation");
  pending[3](cail::GenerationResponse{});
}

class ThrowingMemory : public cail::ConversationMemory {
public:
  bool fail_load{};
  cail::Result<std::vector<cail::Message>> load(const std::string&) override {
    if (fail_load) {
      throw std::runtime_error("load failed");
    }
    return std::vector<cail::Message>{};
  }
  cail::Result<void> append(const std::string&, std::vector<cail::Message>) override {
    throw std::runtime_error("append failed");
  }
  cail::Result<void> clear(const std::string&) override { return {}; }
};

void test_memory_exceptions() {
  auto memory = std::make_shared<ThrowingMemory>();
  auto model = cail::LanguageModel(
      {}, {}, {}, [](const auto&, const auto& complete, const auto&) -> cail::Result<void> {
        complete(cail::GenerationResponse{});
        return {};
      });
  cail::Agent agent({.model = model, .memory = memory, .conversation_id = "same"});
  for (int i = 0; i < 2; ++i) {
    std::promise<cail::Result<cail::GenerationResponse>> completed;
    auto future = completed.get_future();
    check(
        agent.generate_async("hello", [&](auto result) { completed.set_value(std::move(result)); })
            .has_value(),
        "memory-backed generation initiates without waiting for the store");
    auto result = future.get();
    check(!result && result.error().code == cail::ErrorCode::memory,
          "async memory append exceptions reach completion and release the turn");
  }
  memory->fail_load = true;
  std::promise<cail::Result<cail::GenerationResponse>> completed;
  auto future = completed.get_future();
  check(agent.generate_async("hello", [&](auto result) { completed.set_value(std::move(result)); })
            .has_value(),
        "async memory load is admitted before its result is known");
  auto result = future.get();
  check(!result && result.error().code == cail::ErrorCode::memory,
        "memory load exceptions reach completion");
}

void test_concurrent_memory() {
  const auto directory =
      std::filesystem::temp_directory_path() /
      ("cail_async_memory_" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  for (bool file : {false, true}) {
    std::shared_ptr<cail::ConversationMemory> memory =
        file ? std::shared_ptr<cail::ConversationMemory>(
                   std::make_shared<cail::FileConversationMemory>(directory))
             : std::shared_ptr<cail::ConversationMemory>(
                   std::make_shared<cail::InMemoryConversationMemory>());
    auto other = file ? std::shared_ptr<cail::ConversationMemory>(
                            std::make_shared<cail::FileConversationMemory>(directory))
                      : memory;
    std::vector<std::jthread> writers;
    for (int i = 0; i < 8; ++i) {
      writers.emplace_back([&, i] {
        auto store = i % 2 ? memory : other;
        for (int j = 0; j < 12; ++j) {
          check(store->append("same", {cail::Message{.content = {cail::TextPart{.text = "hello"}}}})
                    .has_value(),
                "concurrent memory append succeeds");
          check(store->load("same").has_value(), "concurrent memory load succeeds");
          check(store->clear("unused").has_value(), "concurrent memory clear succeeds");
        }
      });
    }
    writers.clear();
    auto stored = memory->load("same");
    check(stored && stored->size() == 96, "concurrent appends do not race or lose messages");
  }
  std::filesystem::remove_all(directory);
}

struct RetryState {
  std::atomic<int> sends;
  std::atomic<bool> destroyed;
  bool throw_retry{};
};

class RetryTransport : public cail::HttpTransport {
public:
  explicit RetryTransport(std::shared_ptr<RetryState> state) : state_(std::move(state)) {}
  ~RetryTransport() override { state_->destroyed = true; }
  cail::Result<cail::HttpResponse> send(const cail::HttpRequest&, std::stop_token) override {
    return {};
  }
  cail::Result<cail::HttpResponse> stream(const cail::HttpRequest&, const cail::HttpDataHandler&,
                                          std::stop_token) override {
    return {};
  }
  void send_async(cail::HttpRequest, cail::HttpCompletion complete, std::stop_token) override {
    const auto attempt = ++state_->sends;
    if (attempt > 1 && state_->throw_retry) {
      throw std::runtime_error("retry failed to start");
    }
    complete(cail::HttpResponse{.status_code = attempt == 1 ? 503 : 200});
  }

private:
  std::shared_ptr<RetryState> state_;
};

void test_retry_lifetime_and_cancellation() {
  auto state = std::make_shared<RetryState>();
  std::promise<cail::Result<cail::HttpResponse>> completed;
  auto response = completed.get_future();
  {
    cail::RetryingHttpTransport transport(std::make_unique<RetryTransport>(state),
                                          {.max_retries = 1, .initial_delay = 30ms});
    transport.send_async({}, [&](auto result) { completed.set_value(std::move(result)); }, {});
  }
  check(!state->destroyed,
        "pending retry retains its underlying transport after wrapper destruction");
  auto result = response.get();
  check(result && result->status_code == 200 && state->sends == 2,
        "retry safely completes after its wrapper is destroyed");

  state = std::make_shared<RetryState>();
  state->throw_retry = true;
  auto failure = std::make_shared<std::promise<cail::Result<cail::HttpResponse>>>();
  auto failed = failure->get_future();
  {
    cail::RetryingHttpTransport throwing(std::make_unique<RetryTransport>(state),
                                         {.max_retries = 1, .initial_delay = 1ms});
    throwing.send_async({}, [failure](auto result) { failure->set_value(std::move(result)); }, {});
  }
  const auto error = failed.get();
  check(!error && error.error().code == cail::ErrorCode::transport,
        "an exception while starting a retry reaches completion instead of terminating the timer");

  state = std::make_shared<RetryState>();
  cail::RetryingHttpTransport transport(std::make_unique<RetryTransport>(state),
                                        {.max_retries = 1, .initial_delay = 10s});
  std::stop_source stop;
  int callbacks = 0;
  transport.send_async(
      {},
      [&](auto cancelled) {
        ++callbacks;
        check(!cancelled && cancelled.error().code == cail::ErrorCode::cancelled,
              "cancelling retry backoff reports cancellation");
        throw std::runtime_error("cancel callback failed");
      },
      stop.get_token());
  stop.request_stop();
  check(callbacks == 1 && state->sends == 1,
        "retry cancellation is immediate and prevents another attempt");
}

void test_retry_timer_shutdown() {
  bool cancelled = false;
  {
    cail::detail::AsyncRetryTimer timer;
    check(timer.schedule(
              1h, [] { check(false, "shutdown must not start a queued retry"); },
              [&] { cancelled = true; }) != 0,
          "retry timer admits a task");
  }
  check(cancelled, "retry timer shutdown cancels pending operations and joins its worker");
}

} // namespace test

int main() {
  test::test_delayed_tools();
  test::test_tool_cancellation_and_errors();
  test::test_tool_completion_cancellation_race();
  test::test_callbacks_and_initiation();
  test::test_bounded_workers_and_shutdown();
  test::test_streaming();
  test::test_conversation_overlap();
  test::test_memory_exceptions();
  test::test_concurrent_memory();
  test::test_retry_lifetime_and_cancellation();
  test::test_retry_timer_shutdown();
  return test::failures == 0 ? 0 : 1;
}
