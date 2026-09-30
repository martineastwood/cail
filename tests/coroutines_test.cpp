#include "test_support.hpp"

#include <cail/cail.hpp>

#include <thread>

namespace test {

cail::LanguageModel immediate_model() {
  return cail::LanguageModel(
      {}, {}, {},
      [](auto, auto done, auto) -> cail::Result<void> {
        done(cail::GenerationResponse{.text = R"({"city":"Paris"})"});
        done(cail::GenerationResponse{});
        return {};
      },
      [](auto, auto event, auto done, auto) -> cail::Result<void> {
        event(cail::TextDelta{.text = "hello"});
        done(cail::GenerationResponse{.text = "hello"});
        return {};
      });
}

cail::Task<void> generation_calls() {
  auto model = immediate_model();
  const auto response = co_await cail::generate_text_async({.model = model, .prompt = "hello"});
  check(response && response->text == R"({"city":"Paris"})",
        "await text generation without a completion callback");
  const auto invalid = co_await cail::generate_text_async({.model = model});
  check(!invalid && invalid.error().code == cail::ErrorCode::invalid_configuration,
        "initiation errors are returned from co_await");
  const auto direct = co_await model.generate_async(cail::GenerationRequest{});
  check(direct.has_value(), "await low-level model generation");
}

cail::Task<void> object_call() {
  auto model = immediate_model();
  const auto object =
      co_await cail::generate_object_async<Address>({.model = model, .prompt = "city"});
  check(object && object->city == "Paris", "await typed structured output");
}

cail::Task<void> stream_calls() {
  const auto thread = std::this_thread::get_id();
  auto model = immediate_model();
  std::string text;
  auto on_event = [&](const cail::StreamEvent& event) {
    check(std::this_thread::get_id() == thread,
          "coroutine stream events run on the caller executor");
    text += std::get<cail::TextDelta>(event).text;
  };
  const auto streamed =
      co_await cail::stream_text_async({.model = model, .prompt = "hello"}, on_event);
  check(streamed && text == "hello", "await final streamed response after every event");
  text.clear();
  const auto raw_stream = co_await model.stream_async(cail::GenerationRequest{}, on_event);
  check(raw_stream && text == "hello", "await low-level model streaming");
  const auto invalid_stream =
      co_await cail::stream_text_async({.model = model, .prompt = "hello"}, cail::StreamHandler{});
  check(!invalid_stream, "empty coroutine stream handlers remain validation errors");
}

const cail::EmbeddingModel& immediate_embeddings() {
  static const cail::EmbeddingModel embeddings(
      {}, [](auto inputs, auto done, auto) -> cail::Result<void> {
        cail::EmbeddingBatch batch{.dimensions = 2};
        for (const auto& input : inputs)
          batch.embeddings.push_back(
              {.values = {static_cast<float>(input.size()), 1.0F}, .dimensions = 2});
        done(std::move(batch));
        return {};
      });
  return embeddings;
}

cail::Task<void> single_embedding_call() {
  const auto& embeddings = immediate_embeddings();
  const auto embedding = co_await embeddings.embed_async("hello");
  check(embedding && embedding->values[0] == 5.0F, "await a single embedding");
}

cail::Task<void> batch_embedding_call() {
  const auto& embeddings = immediate_embeddings();
  auto inputs = std::vector<std::string>{"a", "bb"};
  const auto batch = co_await embeddings.embed_many_async(std::move(inputs));
  check(batch && batch->embeddings.size() == 2 && batch->embeddings[1].values[0] == 2.0F,
        "await batch embeddings in input order");
}

cail::Task<void> invalid_embedding_call() {
  const auto& embeddings = immediate_embeddings();
  const auto empty = co_await embeddings.embed_async("");
  check(!empty, "await embedding validation errors");
}

cail::Task<void> await_deferred(cail::LanguageModel model) {
  const auto thread = std::this_thread::get_id();
  const auto result = co_await cail::generate_text_async({.model = model, .prompt = "hello"});
  check(result && result->text == "deferred",
        "executor stays alive while native callback work is pending");
  check(std::this_thread::get_id() == thread,
        "foreign-thread completion resumes on the original executor");
}

void test_deferred() {
  std::jthread worker;
  auto model = cail::LanguageModel({}, {}, {}, [&](auto, auto done, auto) -> cail::Result<void> {
    worker = std::jthread([done = std::move(done)] {
      std::this_thread::sleep_for(std::chrono::milliseconds{10});
      done(cail::GenerationResponse{.text = "deferred"});
    });
    return {};
  });
  cail::run(await_deferred(model));
}

cail::Task<void> await_owned(cail::Task<cail::Result<cail::GenerationResponse>> task) {
  const auto result = co_await std::move(task);
  check(result && result->text == R"({"city":"Paris"})",
        "agent coroutine owns its input and agent before it starts");
}

void test_lazy_ownership() {
  std::string prompt = "hello";
  auto model = immediate_model();
  auto agent = std::make_unique<cail::Agent>(cail::AgentConfig{.model = model});
  auto task = agent->generate_async(prompt);
  agent.reset();
  prompt.clear();
  cail::run(await_owned(std::move(task)));

  int requests = 0;
  cail::LanguageModel lazy({}, {}, {}, [&](auto, auto done, auto) -> cail::Result<void> {
    ++requests;
    done(cail::GenerationResponse{});
    return {};
  });
  {
    auto unused = lazy.generate_async(cail::GenerationRequest{});
  }
  check(requests == 0, "unused coroutine tasks do not start requests");
}

cail::Task<void> agent_calls(cail::Agent agent) {
  const auto generated = co_await agent.generate_async("hello");
  check(generated.has_value(), "await agent generation with memory");
  const auto message =
      co_await agent.generate_async(cail::Message{.content = {cail::TextPart{.text = "hello"}}});
  check(message.has_value(), "await agent message generation");
  const auto explicit_request = co_await agent.generate_async(cail::GenerationRequest{});
  check(explicit_request.has_value(), "await explicit agent requests");
  int events = 0;
  const auto streamed = co_await agent.stream_async("hello", [&](const auto&) { ++events; });
  check(streamed && events == 1, "await agent streaming with memory");
  const auto streamed_message = co_await agent.stream_async(
      cail::Message{.content = {cail::TextPart{.text = "hello"}}}, [&](const auto&) { ++events; });
  check(streamed_message.has_value(), "await agent message streaming");
  const auto streamed_request =
      co_await agent.stream_async(cail::GenerationRequest{}, [&](const auto&) { ++events; });
  check(streamed_request.has_value(), "await explicit agent streaming");
}

cail::Task<void> cancelled_call(cail::LanguageModel model) {
  std::stop_source stop;
  stop.request_stop();
  const auto result = co_await cail::generate_text_async(
      {.model = model, .prompt = "hello", .tool_loop = {.stop = stop.get_token()}});
  check(!result && result.error().code == cail::ErrorCode::cancelled,
        "coroutine operations preserve stop-token cancellation");
}

cail::Task<bool> await_external_cancel(cail::LanguageModel model) {
  auto result = co_await model.generate_async(cail::GenerationRequest{});
  co_return !result && result.error().code == cail::ErrorCode::cancelled;
}

void test_executor_cancellation() {
  cail::asio::io_context context;
  cail::asio::cancellation_signal signal;
  cail::LanguageModel::GenerationCompletion pending;
  std::stop_token observed;
  cail::LanguageModel model({}, {}, {}, [&](auto, auto complete, auto stop) -> cail::Result<void> {
    pending = std::move(complete);
    observed = stop;
    return {};
  });
  auto result = cail::asio::co_spawn(
      context, await_external_cancel(model),
      cail::asio::bind_cancellation_slot(signal.slot(), cail::asio::use_future));
  cail::asio::post(context, [&] {
    signal.emit(cail::asio::cancellation_type::terminal);
    check(observed.stop_requested(), "Asio coroutine cancellation reaches native CAIL operations");
    pending(std::unexpected(cail::generation_cancelled_error()));
  });
  context.run();
  check(result.get(),
        "cancelled native operations return their cancellation result to the coroutine");
}

cail::Task<bool> await_rejected(cail::LanguageModel model) {
  const auto result = co_await model.generate_async(cail::GenerationRequest{});
  co_return !result && result.error().code == cail::ErrorCode::transport;
}

void test_rejected_coroutine() {
  cail::LanguageModel::GenerationCompletion retained;
  auto model =
      cail::LanguageModel({}, {}, {}, [&](auto, auto complete, auto) -> cail::Result<void> {
        retained = std::move(complete);
        return std::unexpected(cail::Error{.code = cail::ErrorCode::transport});
      });
  check(cail::run(await_rejected(model)),
        "rejected coroutine operations complete without keeping their executor alive");
  retained(cail::GenerationResponse{});
}

void test_rejected_capture_release() {
  std::function<void(cail::Result<int>)> retained;
  auto capture = std::make_shared<int>(42);
  std::weak_ptr<int> weak = capture;
  const auto started = cail::detail::initiate_async<int>(
      [capture = std::move(capture)](auto) { check(false, "rejected callbacks cannot run"); },
      [&](auto done) -> cail::Result<void> {
        retained = std::move(done);
        return std::unexpected(cail::Error{.code = cail::ErrorCode::transport});
      });
  check(!started && weak.expired(),
        "rejected initiation releases captures even when its callback is retained");
  retained(1);
}

} // namespace test

int main() {
  cail::run(test::generation_calls());
  cail::run(test::object_call());
  cail::run(test::stream_calls());
  cail::run(test::single_embedding_call());
  cail::run(test::batch_embedding_call());
  cail::run(test::invalid_embedding_call());
  test::test_deferred();
  test::test_lazy_ownership();
  auto memory = std::make_shared<cail::InMemoryConversationMemory>();
  cail::run(test::agent_calls(cail::Agent(
      {.model = test::immediate_model(), .memory = memory, .conversation_id = "chat"})));
  test::check(memory->load("chat")->size() == 8,
              "awaited agents store complete turns before returning");
  cail::run(test::cancelled_call(test::immediate_model()));
  test::test_executor_cancellation();
  test::test_rejected_coroutine();
  test::test_rejected_capture_release();
  return test::failures ? 1 : 0;
}
