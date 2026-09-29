#include <cail/cail.hpp>

#include <chrono>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#if defined(__APPLE__)
#include <libproc.h>
#include <unistd.h>
#elif defined(__linux__)
#include <fstream>
#endif

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
  while (status >> name >> value) {
    if (name == "Threads:") {
      size.threads = value;
    }
    if (name == "VmRSS:") {
      size.resident_bytes = value * 1024;
    }
    status.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
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

  std::vector<std::future<cail::Result<cail::GenerationResponse>>> pending;
  const auto async_start = std::chrono::steady_clock::now();
  for (int index = 0; index < 64; ++index) {
    auto completion = std::make_shared<std::promise<cail::Result<cail::GenerationResponse>>>();
    pending.push_back(completion->get_future());
    auto started = model.generate_async(
        prompt(), [completion](cail::Result<cail::GenerationResponse> response) {
          completion->set_value(std::move(response));
        });
    if (!started) {
      return 4;
    }
  }
  const auto launched = std::chrono::steady_clock::now();
  const auto in_flight = process_size();
  for (auto& result : pending) {
    auto response = result.get();
    if (!response || response->text != "Hello") {
      return 5;
    }
  }
  const auto async_end = std::chrono::steady_clock::now();

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
  std::cout << "async_64_launch_ms "
            << std::chrono::duration_cast<milliseconds>(launched - async_start).count() << '\n';
  std::cout << "async_64_total_ms "
            << std::chrono::duration_cast<milliseconds>(async_end - async_start).count() << '\n';
  std::cout << "threads_before " << before.threads << " threads_in_flight " << in_flight.threads
            << '\n';
  std::cout << "resident_before_bytes " << before.resident_bytes << " resident_in_flight_bytes "
            << in_flight.resident_bytes << '\n';
  std::cout << "connection_reused " << connection_reused << '\n';
  return connection_reused ? 0 : 8;
}
