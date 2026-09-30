#include "test_support.hpp"

#include <cail/cail.hpp>

#include <future>
#include <latch>
#include <thread>

namespace test {

void test_scheduling() {
  std::deque<std::function<void()>> tasks;
  cail::AsyncOptions options{.schedule = [&](auto task) { tasks.push_back(std::move(task)); }};
  cail::LanguageModel model(
      {}, {}, {},
      [](auto, auto done, auto) -> cail::Result<void> {
        done(cail::GenerationResponse{.text = "done"});
        done(cail::GenerationResponse{});
        return {};
      },
      [](auto, auto event, auto done, auto stop) -> cail::Result<void> {
        event(cail::TextDelta{.text = "a"});
        event(cail::TextDelta{.text = "b"});
        done(stop.stop_requested() ? cail::Result<cail::GenerationResponse>{std::unexpected(
                                         cail::generation_cancelled_error())}
                                   : cail::GenerationResponse{.text = "ab"});
        return {};
      });
  int completions = 0;
  check(model
            .generate_async(
                {},
                [&](auto result) {
                  check(result && result->text == "done", "scheduled generation preserves result");
                  ++completions;
                },
                {}, options)
            .has_value(),
        "scheduled generation starts");
  check(completions == 0 && tasks.size() == 1,
        "generation completion waits for the application scheduler");
  auto task = std::move(tasks.front());
  tasks.pop_front();
  task();
  check(completions == 1, "duplicate completion schedules only one task");

  std::string order;
  check(model
            .stream_async(
                {}, [&](const auto& event) { order += std::get<cail::TextDelta>(event).text; },
                [&](auto result) {
                  check(result && result->text == "ab",
                        "scheduled stream returns its final response");
                  order += "!";
                },
                {}, options)
            .has_value(),
        "scheduled stream starts");
  check(order.empty() && tasks.size() == 1, "stream events share one serial scheduled task");
  task = std::move(tasks.front());
  tasks.pop_front();
  task();
  check(order == "ab!", "scheduled completion follows every event in order");

  cail::EmbeddingModel embedding({}, [](auto, auto done, auto) -> cail::Result<void> {
    done(cail::EmbeddingBatch{.embeddings = {{.values = {1.0F, 2.0F}, .dimensions = 2}},
                              .dimensions = 2});
    return {};
  });
  bool embedded = false;
  check(static_cast<bool>(embedding) &&
            embedding
                .embed_async(
                    "hello", [&](auto result) { embedded = result && result->dimensions == 2; }, {},
                    options)
                .has_value(),
        "async-only embedding models support callback scheduling");
  check(!embedded && tasks.size() == 1, "embedding completion waits for its scheduler");
  task = std::move(tasks.front());
  tasks.pop_front();
  task();
  check(embedded, "scheduled single embeddings decode their result");

  cail::LanguageModel object_model({}, {}, {}, [](auto, auto done, auto) -> cail::Result<void> {
    done(cail::GenerationResponse{.text = R"({"city":"Paris"})"});
    return {};
  });
  bool decoded = false;
  check(static_cast<bool>(object_model) &&
            cail::generate_object_async<Address>(
                {.model = object_model, .prompt = "Name a city", .tool_loop = {.async = options}},
                [&](auto result) { decoded = result && result->city == "Paris"; })
                .has_value(),
        "typed async generation accepts callback scheduling");
  check(!decoded && tasks.size() == 1, "typed generation waits for scheduled completion");
  task = std::move(tasks.front());
  tasks.pop_front();
  task();
  check(decoded, "scheduled typed generation decodes the requested C++ type");

  options.max_pending_events = 1;
  bool overflow = false;
  check(model
            .stream_async(
                {}, [](const auto&) { check(false, "overflowed queued events must be discarded"); },
                [&](auto result) {
                  overflow = !result && result.error().code == cail::ErrorCode::backpressure;
                },
                {}, options)
            .has_value(),
        "bounded scheduled stream starts");
  task = std::move(tasks.front());
  tasks.pop_front();
  task();
  check(overflow, "a slow consumer receives a backpressure error");

  options.max_pending_events = 256;
  std::stop_source stop;
  bool cancelled = false;
  check(model
            .stream_async(
                {}, [](const auto&) { check(false, "cancelled scheduled events must not run"); },
                [&](auto result) {
                  cancelled = !result && result.error().code == cail::ErrorCode::cancelled;
                },
                stop.get_token(), options)
            .has_value(),
        "scheduled cancellable stream starts");
  stop.request_stop();
  task = std::move(tasks.front());
  tasks.pop_front();
  task();
  check(cancelled, "cancellation suppresses queued events before completion delivery");

  cail::AsyncOptions rejected{
      .schedule = [](auto) { throw std::runtime_error("closed executor"); }};
  bool failed = false;
  check(model.stream_async(
                 {}, [](const auto&) {}, [&](auto result) { failed = !result; }, {}, rejected)
                .has_value() &&
            failed,
        "scheduler rejection reports an error without losing completion");
  failed = false;
  check(model.generate_async(
                 {}, [&](auto result) { failed = !result; }, {}, rejected)
                .has_value() &&
            failed,
        "generation scheduler rejection reports an error");
  check(!model.stream_async(
            {}, [](const auto&) {}, [](auto) {}, {}, {.max_pending_events = 0}),
        "zero event capacity is rejected before initiation");

  cail::StreamHandler late_event;
  cail::LanguageModel rejecting(
      {}, {}, {}, {}, [&](auto, auto event, auto, auto) -> cail::Result<void> {
        late_event = std::move(event);
        return std::unexpected(cail::Error{.code = cail::ErrorCode::transport});
      });
  check(!rejecting.stream_async(
            {}, [](const auto&) { check(false, "rejected stream cannot deliver retained events"); },
            [](auto) {}),
        "native streaming initiation errors return directly");
  late_event(cail::TextDelta{.text = "late"});
}

class DeferredMemory : public cail::ConversationMemory {
public:
  LoadCompletion loaded;
  Completion appended;
  std::vector<cail::Message> stored;
  cail::Result<std::vector<cail::Message>> load(const std::string&) override {
    check(false, "async agents must not call sync memory load");
    return {};
  }
  cail::Result<void> append(const std::string&, std::vector<cail::Message>) override {
    check(false, "async agents must not call sync memory append");
    return {};
  }
  cail::Result<void> clear(const std::string&) override { return {}; }
  cail::Result<void> load_async(std::string, LoadCompletion done, std::stop_token) override {
    loaded = std::move(done);
    return {};
  }
  cail::Result<void> append_async(std::string, std::vector<cail::Message> messages, Completion done,
                                  std::stop_token) override {
    stored = std::move(messages);
    appended = std::move(done);
    return {};
  }
};

void test_agent_streaming_memory() {
  for (int input = 0; input < 3; ++input) {
    auto memory = std::make_shared<DeferredMemory>();
    cail::StreamHandler event;
    cail::LanguageModel::GenerationCompletion done;
    cail::GenerationRequest request;
    cail::LanguageModel model(
        {}, {}, {}, {}, [&](auto value, auto handler, auto complete, auto) -> cail::Result<void> {
          request = std::move(value);
          event = std::move(handler);
          done = std::move(complete);
          return {};
        });
    std::optional<cail::Result<cail::GenerationResponse>> result;
    std::string deltas;
    {
      cail::Agent agent({.model = model,
                         .instructions = "Be concise",
                         .memory = memory,
                         .conversation_id = "chat"});
      auto handler = [&](const auto& value) { deltas += std::get<cail::TextDelta>(value).text; };
      auto complete = [&](auto value) { result = std::move(value); };
      cail::Result<void> started;
      if (input == 0)
        started = agent.stream_async("hello", handler, complete);
      else if (input == 1)
        started = agent.stream_async(cail::Message{.content = {cail::TextPart{.text = "hello"}}},
                                     handler, complete);
      else
        started = agent.stream_async(
            cail::GenerationRequest{
                .messages = {cail::Message{.content = {cail::TextPart{.text = "hello"}}}}},
            handler, complete);
      check(started.has_value(), "every agent async streaming overload starts");
      if (input != 2)
        check(!done && memory->loaded && !result, "agent returns while remote memory is pending");
    }
    if (input != 2) {
      auto loaded = std::move(memory->loaded);
      loaded(std::vector<cail::Message>{
          cail::Message{.content = {cail::TextPart{.text = "earlier"}}}});
      check(request.messages.size() == 3, "agent stream includes instructions and loaded history");
    } else {
      check(!memory->loaded && request.messages.size() == 2,
            "explicit agent requests bypass memory");
    }
    event(cail::TextDelta{.text = "hi"});
    auto complete = std::move(done);
    complete(cail::GenerationResponse{.text = "hi"});
    if (input != 2) {
      check(!result && memory->appended && memory->stored.size() == 2,
            "completion waits for remote memory to persist the whole turn");
      auto appended = std::move(memory->appended);
      appended({});
    }
    check(result && *result && deltas == "hi",
          "async agent survives destruction and delivers its stream");
  }
}

void test_agent_memory_cancellation() {
  auto memory = std::make_shared<DeferredMemory>();
  int requests = 0;
  cail::LanguageModel model({}, {}, {}, [&](auto, auto done, auto) -> cail::Result<void> {
    ++requests;
    done(cail::GenerationResponse{});
    return {};
  });
  cail::Agent agent({.model = model, .memory = memory, .conversation_id = "chat"});
  std::stop_source stop;
  bool cancelled = false;
  check(agent
            .generate_async("hello",
                            [&](auto result) {
                              cancelled =
                                  !result && result.error().code == cail::ErrorCode::cancelled;
                            },
                            {.stop = stop.get_token()})
            .has_value(),
        "agent starts with a pending memory load");
  stop.request_stop();
  auto loaded = std::move(memory->loaded);
  loaded(std::vector<cail::Message>{});
  check(cancelled && requests == 0 && !memory->appended,
        "cancelled memory load prevents model requests and writes");
  check(agent.generate_async("next", [](auto) {}).has_value(),
        "cancelled memory operation releases its conversation turn");
  loaded = std::move(memory->loaded);
  loaded(std::unexpected(cail::Error{.code = cail::ErrorCode::memory}));
}

class BlockingMemory : public cail::ConversationMemory {
public:
  std::latch entered{1};
  std::promise<void> release;
  std::shared_future<void> released{release.get_future()};
  cail::Result<std::vector<cail::Message>> load(const std::string&) override {
    entered.count_down();
    released.wait();
    return std::vector<cail::Message>{};
  }
  cail::Result<void> append(const std::string&, std::vector<cail::Message>) override { return {}; }
  cail::Result<void> clear(const std::string&) override { return {}; }
};

void test_blocking_memory_adapter() {
  auto memory = std::make_shared<BlockingMemory>();
  std::promise<cail::Result<std::vector<cail::Message>>> completed;
  auto result = completed.get_future();
  check(memory->load_async("chat", [&](auto value) { completed.set_value(std::move(value)); })
            .has_value(),
        "blocking stores initiate on bounded memory workers");
  memory->entered.wait();
  check(result.wait_for(std::chrono::milliseconds{0}) == std::future_status::timeout,
        "a blocked memory load leaves the calling thread free");
  memory->release.set_value();
  check(result.get().has_value(), "blocking memory adapter eventually completes");
}

class StreamingRetryTransport : public cail::HttpTransport {
public:
  std::atomic<int> attempts{};
  bool emit_before_failure{};
  cail::Result<cail::HttpResponse> send(const cail::HttpRequest&, std::stop_token) override {
    return {};
  }
  cail::Result<cail::HttpResponse> stream(const cail::HttpRequest&, const cail::HttpDataHandler&,
                                          std::stop_token) override {
    return {};
  }
  void stream_async(cail::HttpRequest, cail::HttpDataHandler data, cail::HttpCompletion done,
                    std::stop_token) override {
    if (++attempts == 1) {
      if (emit_before_failure)
        data("partial");
      done(cail::HttpResponse{.status_code = 503});
    } else {
      data("done");
      done(cail::HttpResponse{.status_code = 200});
    }
  }
};

void test_stream_retries() {
  for (bool emitted : {false, true}) {
    auto transport = std::make_unique<StreamingRetryTransport>();
    auto* underlying = transport.get();
    underlying->emit_before_failure = emitted;
    cail::RetryingHttpTransport retry(std::move(transport),
                                      {.initial_delay = std::chrono::milliseconds{1}});
    std::promise<cail::Result<cail::HttpResponse>> completed;
    auto result = completed.get_future();
    std::string bytes;
    retry.stream_async(
        {}, [&](auto data) { bytes += data; },
        [&](auto response) { completed.set_value(std::move(response)); }, {});
    auto response = result.get();
    check(response && response->status_code == (emitted ? 503 : 200) &&
              underlying->attempts == (emitted ? 1 : 2),
          "async streaming retries HTTP failures only before any delivered bytes");
    check(bytes == (emitted ? "partial" : "done"), "stream retries never replay delivered data");
  }
}

} // namespace test

int main() {
  test::test_scheduling();
  test::test_agent_streaming_memory();
  test::test_agent_memory_cancellation();
  test::test_blocking_memory_adapter();
  test::test_stream_retries();
  return test::failures ? 1 : 0;
}
