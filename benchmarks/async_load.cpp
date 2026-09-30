#include <cail/cail.hpp>

#include <chrono>
#include <future>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#include <libproc.h>
#include <unistd.h>
#elif defined(__linux__)
#include <fstream>
#endif

struct CountInput {
  std::string query;
};

namespace {

struct ProcessSize {
  std::size_t threads{};
  std::size_t resident_bytes{};
};

ProcessSize process_size() {
#if defined(__APPLE__)
  proc_taskinfo info{};
  if (proc_pidinfo(getpid(), PROC_PIDTASKINFO, 0, &info, sizeof(info)) == sizeof(info)) {
    return {.threads = static_cast<std::size_t>(info.pti_threadnum),
            .resident_bytes = static_cast<std::size_t>(info.pti_resident_size)};
  }
#elif defined(__linux__)
  std::ifstream status("/proc/self/status");
  ProcessSize size;
  std::string name;
  std::size_t value;
  std::string line;
  while (std::getline(status, line)) {
    std::istringstream fields(line);
    if (!(fields >> name >> value))
      continue;
    if (name == "Threads:") {
      size.threads = value;
    }
    if (name == "VmRSS:") {
      size.resident_bytes = value * 1024;
    }
  }
  return size;
#endif
  return {};
}

cail::GenerationRequest prompt() {
  return {.messages = {cail::Message{.content = {cail::TextPart{.text = "Hello"}}}}};
}

std::string connection_id(const cail::HttpResponse& response) {
  for (const auto& header : response.headers) {
    if (header.name == "X-Connection-Id") {
      return header.value;
    }
  }
  return {};
}

template <typename Start> bool measure(std::string_view name, int count, Start start) {
  auto peak = process_size();
  std::jthread sampler([&](std::stop_token stop) {
    while (!stop.stop_requested()) {
      const auto size = process_size();
      peak.threads = std::max(peak.threads, size.threads);
      peak.resident_bytes = std::max(peak.resident_bytes, size.resident_bytes);
      std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
  });
  std::vector<std::future<cail::Result<cail::GenerationResponse>>> pending;
  const auto began = std::chrono::steady_clock::now();
  for (int i = 0; i < count; ++i) {
    auto completion = std::make_shared<std::promise<cail::Result<cail::GenerationResponse>>>();
    pending.push_back(completion->get_future());
    if (!start([completion](auto response) { completion->set_value(std::move(response)); }))
      return false;
  }
  const auto launched = std::chrono::steady_clock::now();
  for (auto& response : pending)
    if (!response.get())
      return false;
  const auto finished = std::chrono::steady_clock::now();
  sampler.request_stop();
  sampler.join();
  using milliseconds = std::chrono::milliseconds;
  std::cout << name << " launch_ms "
            << std::chrono::duration_cast<milliseconds>(launched - began).count() << " total_ms "
            << std::chrono::duration_cast<milliseconds>(finished - began).count()
            << " peak_threads " << peak.threads << " peak_resident_bytes " << peak.resident_bytes
            << '\n';
  return true;
}

} // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    return 2;
  }
  const std::string base = argv[1];
  auto model = cail::create_local({.endpoint = base + "/chat/delay"})("test-model");
  const auto before = process_size();

  const auto blocking_start = std::chrono::steady_clock::now();
  for (int index = 0; index < 4; ++index) {
    auto response = model.generate(prompt());
    if (!response || response->text != "Hello") {
      return 3;
    }
  }
  const auto blocking_end = std::chrono::steady_clock::now();

  if (!measure("model_async_64", 64,
               [&](auto complete) { return model.generate_async(prompt(), std::move(complete)); }))
    return 4;
  if (!measure("text_async_64", 64, [&](auto complete) {
        return cail::generate_text_async({.model = model, .prompt = "Hello"}, std::move(complete));
      }))
    return 5;

  auto tool_model = cail::create_local({.endpoint = base + "/async/tool-chat"})("test-model");
  auto count = cail::tool<CountInput, int>("count", "Count", [](const CountInput& input) {
    std::this_thread::sleep_for(std::chrono::milliseconds{20});
    return static_cast<int>(input.query.size());
  });
  if (!measure("tools_async_64", 64, [&](auto complete) {
        return cail::generate_text_async(
            {.model = tool_model, .prompt = "count abc", .tools = {count}}, std::move(complete));
      }))
    return 5;

  auto stream_model = cail::create_local({.endpoint = base + "/chat"})("test-model");
  if (!measure("streams_async_64", 64, [&](auto complete) {
        return cail::stream_text_async(
            {.model = stream_model, .prompt = "Hello"}, [](const auto&) {}, std::move(complete));
      }))
    return 5;

  auto transport = cail::make_default_http_transport();
  std::string first_connection;
  bool connection_reused = true;
  for (int index = 0; index < 3; ++index) {
    auto response = transport->send({.url = base + "/chat", .body = "{}"}, {});
    if (!response || response->status_code != 200) {
      return 6;
    }
    const auto id = connection_id(*response);
    if (id.empty()) {
      return 7;
    }
    if (index == 0) {
      first_connection = id;
    } else {
      connection_reused &= id == first_connection;
    }
  }

  using milliseconds = std::chrono::milliseconds;
  std::cout << "blocking_4_ms "
            << std::chrono::duration_cast<milliseconds>(blocking_end - blocking_start).count()
            << '\n';
  std::cout << "threads_before " << before.threads << " resident_before_bytes "
            << before.resident_bytes << '\n';
  std::cout << "connection_reused " << connection_reused << '\n';
  return connection_reused ? 0 : 8;
}
