#include "test_support.hpp"

#include <future>

#include <cail/cail.hpp>

#include <array>
#include <limits>

namespace test {

using Family = cail::OpenCodeApiFamily;
constexpr std::array families{Family::responses, Family::chat_completions,
                              Family::anthropic_messages, Family::gemini};

std::string response_body(Family family, std::string_view reason) {
  switch (family) {
  case Family::responses:
    return "{\"status\":\"" + std::string{reason == "completed" ? "completed" : "incomplete"} +
           "\",\"output\":[],\"incomplete_details\":{\"reason\":\"" + std::string{reason} + "\"}}";
  case Family::chat_completions:
    return "{\"choices\":[{\"index\":0,\"message\":{\"content\":\"ok\"},\"finish_reason\":\"" +
           std::string{reason} + "\"}]}";
  case Family::anthropic_messages:
    return "{\"content\":[],\"stop_reason\":\"" + std::string{reason} + "\"}";
  case Family::gemini:
    return "{\"candidates\":[{\"finishReason\":\"" + std::string{reason} + "\"}]}";
  }
  std::unreachable();
}

cail::GenerationRequest request() {
  return {.messages = {cail::Message{.content = {cail::TextPart{.text = "Hello"}}}},
          .tools = {cail::make_tool<ToolInput>("count", "Count characters")},
          .session_id = "test-session",
          .max_output_tokens = 100,
          .temperature = 0.4,
          .top_p = 0.8};
}

void test_encoding_and_validation() {
  for (const auto family : families) {
    auto transport = std::make_unique<StubTransport>();
    auto* stub = transport.get();
    stub->response.body = response_body(family, family == Family::responses            ? "completed"
                                                : family == Family::gemini             ? "STOP"
                                                : family == Family::anthropic_messages ? "end_turn"
                                                                                       : "stop");
    auto model = cail::create_opencode({.api_key = "key"})("model", family, std::move(transport));
    for (const auto mode : {cail::ToolChoiceMode::auto_, cail::ToolChoiceMode::none,
                            cail::ToolChoiceMode::required, cail::ToolChoiceMode::named}) {
      auto input = request();
      input.tool_choice = cail::ToolChoice{
          .mode = mode, .name = mode == cail::ToolChoiceMode::named ? "count" : ""};
      if (family != Family::responses) {
        input.stop_sequences = {"END"};
      }
      check(model.generate(input).has_value(), "all families accept portable controls");
      check(stub->request.body.find("\"temperature\":0.4") != std::string::npos &&
                stub->request.body.find(family == Family::gemini
                                            ? "\"topP\":0.8"
                                            : "\"top_p\":0.8") != std::string::npos,
            "sampling controls reach each provider");
      if (family != Family::responses) {
        check(stub->request.body.find("[\"END\"]") != std::string::npos,
              "supported adapters encode stop sequences");
      }
      const std::string expected =
          family == Family::gemini ? (mode == cail::ToolChoiceMode::none    ? "NONE"
                                      : mode == cail::ToolChoiceMode::auto_ ? "AUTO"
                                                                            : "ANY")
          : mode == cail::ToolChoiceMode::required
              ? (family == Family::anthropic_messages ? "any" : "required")
          : mode == cail::ToolChoiceMode::named
              ? (family == Family::anthropic_messages ? "tool" : "function")
          : mode == cail::ToolChoiceMode::none ? "none"
                                               : "auto";
      check(stub->request.body.find('"' + expected + '"') != std::string::npos,
            "tool choice uses the provider's mode");
      if (mode == cail::ToolChoiceMode::named) {
        check(stub->request.body.find(family == Family::gemini
                                          ? "\"allowedFunctionNames\":[\"count\"]"
                                          : "\"name\":\"count\"") != std::string::npos,
              "named tool choice reaches the provider");
      }
    }
    const auto sent = stub->send_count;
    auto invalid = request();
    const auto reject = [&](const cail::GenerationRequest& input) {
      const auto result = model.generate(input);
      check(!result && result.error().code == cail::ErrorCode::invalid_configuration &&
                stub->send_count == sent,
            "invalid controls fail before HTTP");
    };
    invalid.temperature = std::numeric_limits<double>::quiet_NaN();
    reject(invalid);
    invalid = request();
    invalid.top_p = 1.1;
    reject(invalid);
    invalid = request();
    invalid.tool_choice = cail::ToolChoice{.mode = cail::ToolChoiceMode::named, .name = "missing"};
    reject(invalid);
    invalid = request();
    invalid.provider_options = {{"temperature", 0.2}};
    reject(invalid);
    invalid = request();
    invalid.stop_sequences = {""};
    reject(invalid);
    if (family == Family::responses) {
      invalid.stop_sequences = {"END"};
      reject(invalid);
    }
  }
}

void test_finish_reasons() {
  for (const auto family : families) {
    const auto reasons = family == Family::responses ? std::array{"completed", "max_output_tokens",
                                                                  "content_filter", "future_reason"}
                         : family == Family::anthropic_messages
                             ? std::array{"end_turn", "max_tokens", "refusal", "future_reason"}
                         : family == Family::gemini
                             ? std::array{"STOP", "MAX_TOKENS", "SAFETY", "future_reason"}
                             : std::array{"stop", "length", "content_filter", "future_reason"};
    const std::array expected{cail::FinishReason::stop, cail::FinishReason::length,
                              cail::FinishReason::content_filter, cail::FinishReason::other};
    for (std::size_t i = 0; i < reasons.size(); ++i) {
      auto transport = std::make_unique<StubTransport>();
      const auto body = response_body(family, reasons[i]);
      transport->response.body = body;
      if (family == Family::responses) {
        transport->chunks = {"data: {\"type\":\"response." +
                             std::string{i == 0 ? "completed" : "incomplete"} +
                             "\",\"response\":" + body + "}\n\n"};
      } else if (family == Family::anthropic_messages) {
        transport->chunks = {
            "data: {\"type\":\"message_start\",\"message\":{\"content\":[]}}\n\n",
            "data: "
            "{\"type\":\"message_delta\",\"delta\":{\"type\":\"message_delta\",\"stop_reason\":\"" +
                std::string{reasons[i]} + "\"}}\n\n",
            "data: {\"type\":\"message_stop\"}\n\n"};
      } else if (family == Family::gemini) {
        transport->chunks = {"data: " + body + "\n\n"};
      } else {
        transport->chunks = {"data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"" +
                                 std::string{reasons[i]} + "\"}]}\n\n",
                             "data: [DONE]\n\n"};
      }
      auto model = cail::create_opencode({.api_key = "key"})("model", family, std::move(transport));
      for (const auto streaming : {false, true}) {
        const auto result = streaming ? model.stream(request(), [](const cail::StreamEvent&) {})
                                      : model.generate(request());
        check(result && result->finish_reason == expected[i] &&
                  result->raw_finish_reason == reasons[i],
              "blocking and streaming normalize and retain finish reasons");
        check(result && result->status == (i == 0   ? cail::GenerationStatus::completed
                                           : i == 2 ? cail::GenerationStatus::refused
                                                    : cail::GenerationStatus::incomplete),
              "finish reasons set consistent generation status");
      }
    }
  }
}

void test_step_usage() {
  const auto count = cail::tool<ToolInput, ToolOutput>(
      "count", "Count", [](const ToolInput&) { return ToolOutput{.count = 3}; });
  for (const auto execution : {0, 1, 2}) {
    ScriptedClient client({
        {.text = "checking",
         .finish_reason = cail::FinishReason::tool_calls,
         .usage = cail::TokenUsage{.input_tokens = 10, .output_tokens = 2, .cache_read_tokens = 4},
         .tool_calls = {{.id = "call-1", .name = "count", .arguments = R"({"query":"abc"})"}}},
        {.finish_reason = cail::FinishReason::tool_calls,
         .tool_calls = {{.id = "call-2", .name = "count", .arguments = R"({"query":"abc"})"}}},
        {.text = "done",
         .finish_reason = cail::FinishReason::stop,
         .usage = cail::TokenUsage{.input_tokens = 5, .output_tokens = 3, .reasoning_tokens = 1}},
    });
    cail::LanguageModel model(
        [&](const cail::GenerationRequest& input, const std::stop_token& stop) {
          return client.generate(input, stop);
        },
        [&](const cail::GenerationRequest& input, const cail::StreamHandler& handler,
            const std::stop_token& stop) { return client.stream(input, handler, stop); },
        {},
        [&](const cail::GenerationRequest& input,
            const cail::LanguageModel::GenerationCompletion& complete,
            const std::stop_token& stop) -> cail::Result<void> {
          complete(client.generate(input, stop));
          return {};
        });
    cail::GenerateTextOptions options{
        .model = model,
        .prompt = "count",
        .tools = {count},
        .max_output_tokens = 100,
        .temperature = 0.4,
        .top_p = 0.8,
        .stop_sequences = {"END"},
        .tool_choice = cail::ToolChoice{.mode = cail::ToolChoiceMode::named, .name = "count"}};
    std::optional<cail::Result<cail::GenerationResponse>> response;
    if (execution == 0) {
      response = cail::generate_text(options);
    } else if (execution == 1) {
      response = cail::stream_text(options, [](const cail::StreamEvent&) {});
    } else {
      std::promise<cail::Result<cail::GenerationResponse>> completed;
      auto pending = completed.get_future();
      check(cail::generate_text_async(options,
                                      [&](cail::Result<cail::GenerationResponse> result) {
                                        completed.set_value(std::move(result));
                                      })
                .has_value(),
            "async tool loop starts");
      response = pending.get();
    }
    check(response && *response && (*response)->steps.size() == 3 && (*response)->usage &&
              (*response)->usage->input_tokens == 5 && (*response)->total_usage &&
              (*response)->total_usage->input_tokens == 15 &&
              (*response)->total_usage->output_tokens == 5 &&
              (*response)->total_usage->cache_read_tokens == 4 &&
              (*response)->total_usage->reasoning_tokens == 1 &&
              !(*response)->total_usage->cache_write_tokens && !(*response)->steps[1].usage &&
              (*response)->steps[2].step == 2,
          "sync, streaming, and async retain steps and sum only reported usage");
    check(client.requests.size() == 3 && client.requests[0].temperature == 0.4 &&
              client.requests[0].tool_choice->mode == cail::ToolChoiceMode::named &&
              client.requests[1].tool_choice->mode == cail::ToolChoiceMode::auto_ &&
              client.requests[2].top_p == 0.8 &&
              client.requests[2].stop_sequences == std::vector<std::string>{"END"} &&
              client.requests[2].max_output_tokens == 100,
          "high-level functions preserve controls and release forced choice after tools return");
  }
  cail::TokenUsage usage;
  cail::detail::anthropic::apply_usage(usage, {.input_tokens = 5,
                                               .output_tokens = 2,
                                               .cache_read_input_tokens = 10,
                                               .cache_creation_input_tokens = 3});
  cail::detail::anthropic::apply_usage(usage, {.output_tokens = 4});
  check(usage.input_tokens == 18 && usage.output_tokens == 4,
        "Anthropic includes cache reads and writes without double counting stream updates");
  cail::GenerationResponse response;
  cail::detail::gemini::apply_usage(
      response, {.promptTokenCount = 5, .candidatesTokenCount = 2, .thoughtsTokenCount = 3});
  cail::detail::gemini::apply_usage(response, {.candidatesTokenCount = 4});
  check(response.usage && response.usage->input_tokens == 5 && response.usage->output_tokens == 7,
        "Gemini includes reasoning without double counting stream updates");
}

} // namespace test

int main() {
  test::test_encoding_and_validation();
  test::test_finish_reasons();
  test::test_step_usage();
  return test::failures == 0 ? 0 : 1;
}
