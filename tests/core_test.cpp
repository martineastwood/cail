#include "test_support.hpp"

#include <cail/agent.hpp>
#include <cail/detail/strict_schema.hpp>
#include <cail/field.hpp>
#include <cail/generate.hpp>
#include <cail/json.hpp>
#include <cail/schema.hpp>
#include <cail/tool.hpp>

#include <algorithm>
#include <cstdint>
#include <future>
#include <memory>

namespace test {

enum class ValidationChoice { Yes, No };

struct Summary {
  cail::Field<std::string> title{.min_length = 1, .max_length = 2};
  cail::Field<std::vector<std::string>> highlights{.min_items = 1, .max_items = 2};
};

struct NestedSummary {
  std::vector<Summary> summaries;
  std::optional<Summary> extra;
};

struct EmptySummary {
  cail::Field<std::string> title{.min_length = 0, .max_length = 0};
  cail::Field<std::vector<std::string>> highlights{.min_items = 0, .max_items = 0};
};

void test_length_constraints() {
  check(cail::from_json<EmptySummary>(R"({"title":"","highlights":[]})").has_value() &&
            !cail::from_json<EmptySummary>(R"({"title":"a","highlights":[]})") &&
            !cail::from_json<EmptySummary>(R"({"title":"","highlights":["one"]})"),
        "zero length bounds are enforced rather than treated as absent");
  check(cail::from_json<cail::Field<std::string>>(R"("unrestricted")").has_value() &&
            cail::from_json<cail::Field<std::vector<int>>>("[]").has_value(),
        "omitted length bounds leave strings and arrays unrestricted");
  const auto schema = cail::schema<Summary>();
  const auto& title = *schema.properties->at("title");
  const auto& highlights = *schema.properties->at("highlights");
  check(title.min_length == 1 && title.max_length == 2 && highlights.min_items == 1 &&
            highlights.max_items == 2,
        "typed fields emit string and array length constraints");
  const auto encoded = cail::json_schema<Summary>();
  const auto decoded = cail::schema_from_json(*encoded);
  check(decoded && decoded->properties->at("title")->min_length == 1 &&
            decoded->properties->at("title")->max_length == 2 &&
            decoded->properties->at("highlights")->min_items == 1 &&
            decoded->properties->at("highlights")->max_items == 2,
        "length constraints survive JSON Schema round trips");
  const auto strict = cail::detail::strict_json_schema(schema);
  check(strict && strict->properties->at("title")->max_length == 2 &&
            strict->properties->at("highlights")->max_items == 2,
        "strict provider schemas preserve length constraints");
  for (const auto json :
       {R"({"title":"a","highlights":["one"]})", R"({"title":"ab","highlights":["one","two"]})",
        R"({"title":"é😀","highlights":["one"]})",
        R"({"title":"\u00e9\ud83d\ude00","highlights":["one"]})",
        R"({"title":"e\u0301","highlights":["one"]})",
        R"({"title":"\u0000","highlights":["one"]})"})
    check(cail::from_json<Summary>(json).has_value(),
          "inclusive string and array bounds count Unicode code points");
  const std::vector<std::pair<std::string, std::string>> invalid{
      {R"({"title":"","highlights":["one"]})", "$.title: value does not satisfy min_length."},
      {R"({"title":"abc","highlights":["one"]})", "$.title: value does not satisfy max_length."},
      {R"({"title":"a","highlights":[]})", "$.highlights: value does not satisfy min_items."},
      {R"({"title":"a","highlights":["one","two","three"]})",
       "$.highlights: value does not satisfy max_items."},
  };
  for (const auto& [json, message] : invalid) {
    const auto result = cail::from_json<Summary>(json);
    check(!result && result.error().code == cail::ErrorCode::schema_validation &&
              result.error().message == message,
          "length validation identifies the field and failed constraint");
    cail::LanguageModel model(
        [json](const cail::GenerationRequest&,
               std::stop_token) -> cail::Result<cail::GenerationResponse> {
          return cail::GenerationResponse{.text = json};
        },
        {}, {},
        [json](cail::GenerationRequest, auto complete, std::stop_token) -> cail::Result<void> {
          complete(cail::GenerationResponse{.text = json});
          return {};
        });
    const auto generated = cail::generate_object<Summary>({.model = model, .prompt = "summarize"});
    check(!generated && generated.error().message == message,
          "generated objects enforce length constraints");
    std::promise<cail::Result<Summary>> completion;
    auto future = completion.get_future();
    const auto started = cail::generate_object_async<Summary>(
        {.model = model, .prompt = "summarize"},
        [&](auto output) { completion.set_value(std::move(output)); });
    check(started.has_value(), "async length-constrained generation starts");
    if (started) {
      const auto output = future.get();
      check(!output && output.error().message == message,
            "async generated objects enforce length constraints");
    }
    bool invoked = false;
    auto tool = cail::tool<Summary, Summary>("summary", "Summary", [&](const Summary& input) {
      invoked = true;
      return input;
    });
    const cail::ToolCall call{.name = "summary", .arguments = json};
    const auto output = tool.execute(call, {});
    check(!output && output.error().message == message && !invoked,
          "invalid lengths never reach synchronous tool handlers");
    auto async = cail::async_tool<Summary, Summary>(
        "summary", "Summary", [&](Summary input, cail::ToolContext, auto complete) {
          invoked = true;
          complete(std::move(input));
        });
    std::promise<cail::Result<std::string>> tool_completion;
    auto tool_future = tool_completion.get_future();
    const auto tool_started = async.execute_async(
        call, {}, [&](auto value) { tool_completion.set_value(std::move(value)); });
    check(tool_started.has_value(), "async length-constrained tool starts");
    if (tool_started) {
      const auto value = tool_future.get();
      check(!value && value.error().message == message && !invoked,
            "invalid lengths never reach asynchronous tool handlers");
    }
  }
  const auto nested =
      cail::from_json<NestedSummary>(R"({"summaries":[{"title":"abc","highlights":["one"]}]})");
  check(!nested && nested.error().message.starts_with("$.summaries[0].title:"),
        "nested length errors include the array index");
  const auto optional =
      cail::from_json<NestedSummary>(R"({"summaries":[],"extra":{"title":"a","highlights":[]}})");
  check(!optional && optional.error().message.starts_with("$.extra.highlights:"),
        "optional nested values enforce array lengths");
  for (const auto keyword : {"minLength", "maxLength"})
    for (const auto value : {"-1", "1.5", "1e100", "null", "true", R"("2")"})
      check(!cail::schema_from_json(std::string{R"({"type":"string",")"} + keyword + "\":" + value +
                                    "}"),
            "schema length constraints require nonnegative integers");
}

struct BoundedValue {
  cail::Field<double> score{.minimum = 0.0, .maximum = 1.0};
};

struct NestedBounds {
  cail::Field<std::vector<BoundedValue>> values;
  std::optional<BoundedValue> extra;
};

struct BoundedInteger {
  cail::Field<std::uint64_t> value{.minimum = 9007199254740993ULL, .maximum = 9007199254740994ULL};
};

void test_constraint_validation() {
  check(cail::from_json<BoundedInteger>(R"({"value":9007199254740993})").has_value() &&
            cail::from_json<BoundedInteger>(R"({"value":9007199254740994})").has_value() &&
            !cail::from_json<BoundedInteger>(R"({"value":9007199254740992})") &&
            !cail::from_json<BoundedInteger>(R"({"value":9007199254740995})"),
        "integer bounds retain precision beyond the exact range of double");
  check(cail::from_json<double>("2").has_value(), "unconstrained numbers remain unrestricted");
  for (const auto json : {R"({"score":0})", R"({"score":1})", R"({"score":0.5})"})
    check(cail::from_json<BoundedValue>(json).has_value(), "numeric bounds are inclusive");
  for (const auto json : {R"({"score":-0.1})", R"({"score":1.1})"}) {
    const auto result = cail::from_json<BoundedValue>(json);
    check(!result && result.error().code == cail::ErrorCode::schema_validation &&
              result.error().message.starts_with("$.score:") &&
              result.error().message.find(json == std::string_view{R"({"score":-0.1})"}
                                              ? "minimum"
                                              : "maximum") != std::string::npos,
          "out-of-range values report the field and failed constraint");
  }
  check(cail::from_json<NestedBounds>(R"({"values":[{"score":0.5}],"extra":null})").has_value(),
        "valid nested values and null optionals decode");
  const auto array = cail::from_json<NestedBounds>(R"({"values":[{"score":0.5},{"score":2}]})");
  check(!array && array.error().message.starts_with("$.values[1].score:"),
        "array validation reports the failing element");
  const auto optional = cail::from_json<NestedBounds>(R"({"values":[],"extra":{"score":-1}})");
  check(!optional && optional.error().message.starts_with("$.extra.score:"),
        "populated optionals validate nested fields");
  check(cail::from_json<ValidationChoice>(R"("Yes")").has_value() &&
            !cail::from_json<ValidationChoice>(R"("Maybe")"),
        "enum decoding rejects unknown choices");

  cail::LanguageModel model(
      [](const cail::GenerationRequest&,
         std::stop_token) -> cail::Result<cail::GenerationResponse> {
        return cail::GenerationResponse{.text = R"({"score":2})"};
      },
      {}, {},
      [](cail::GenerationRequest, cail::LanguageModel::GenerationCompletion complete,
         std::stop_token) -> cail::Result<void> {
        complete(cail::GenerationResponse{.text = R"({"score":2})"});
        return {};
      });
  const auto generated = cail::generate_object<BoundedValue>({.model = model, .prompt = "score"});
  check(!generated && generated.error().code == cail::ErrorCode::schema_validation,
        "generated objects enforce numeric constraints");
  std::promise<cail::Result<BoundedValue>> completed;
  auto future = completed.get_future();
  const auto started = cail::generate_object_async<BoundedValue>(
      {.model = model, .prompt = "score"},
      [&](cail::Result<BoundedValue> value) { completed.set_value(std::move(value)); });
  check(started.has_value(), "async constrained generation starts");
  if (started) {
    const auto result = future.get();
    check(!result && result.error().code == cail::ErrorCode::schema_validation,
          "async generated objects enforce numeric constraints");
  }

  bool invoked = false;
  auto tool =
      cail::tool<BoundedValue, BoundedValue>("score", "Score", [&](const BoundedValue& input) {
        invoked = true;
        return input;
      });
  const cail::ToolCall call{.id = "invalid", .name = "score", .arguments = R"({"score":2})"};
  const auto result = tool.execute(call, {});
  check(!result && result.error().code == cail::ErrorCode::schema_validation && !invoked,
        "invalid tool arguments never reach synchronous handlers");
  auto async = cail::async_tool<BoundedValue, BoundedValue>(
      "score", "Score", [&](BoundedValue input, cail::ToolContext, auto complete) {
        invoked = true;
        complete(std::move(input));
      });
  std::promise<cail::Result<std::string>> tool_completed;
  auto tool_future = tool_completed.get_future();
  const auto tool_started = async.execute_async(
      call, {}, [&](auto output) { tool_completed.set_value(std::move(output)); });
  check(tool_started.has_value(), "async constrained tool starts");
  if (tool_started) {
    const auto output = tool_future.get();
    check(!output && output.error().code == cail::ErrorCode::schema_validation && !invoked,
          "invalid tool arguments never reach asynchronous handlers");
  }
}

void test_control_character_json() {
  std::string controls;
  for (int ch = 0; ch < 32; ++ch) {
    controls += static_cast<char>(ch);
  }
  const auto encoded = cail::to_json(Address{.city = controls});
  check(encoded.has_value(), "control characters serialize");
  if (!encoded) {
    return;
  }
  const auto decoded = cail::from_json<Address>(*encoded);
  check(decoded && decoded->city == controls, "JSON preserves every control character");
  const auto merged = cail::merge_json_objects(*encoded, {{"extra", std::string(1, '\0')}});
  check(merged.has_value(), "provider options merge with control characters");
  if (!merged) {
    return;
  }
  glz::generic parsed;
  const auto error = glz::read_json(parsed, *merged);
  check(!error && parsed["city"].get<std::string>() == controls &&
            parsed["extra"].get<std::string>() == std::string(1, '\0'),
        "merged JSON preserves control characters in both objects");
}

void test_additional_properties_schema() {
  for (const auto value : {"false", "true", "{}", R"({"type":"string"})",
                           R"({"anyOf":[{"type":"string"},{"type":"number"}]})"}) {
    const auto json = std::string{R"({"type":"object","additionalProperties":)"} + value + "}";
    const auto schema = cail::schema_from_json(json);
    check(schema.has_value(), "additionalProperties accepts booleans and schema objects");
    if (!schema) {
      continue;
    }
    const auto encoded = cail::to_json(*schema);
    const auto decoded = encoded ? cail::schema_from_json(*encoded) : cail::Result<cail::Schema>{};
    check(decoded && decoded->additional_properties &&
              cail::to_json(*decoded->additional_properties) == std::string{value},
          "additionalProperties round-trips without losing schema constraints");
    const auto strict = cail::detail::strict_json_schema(*schema);
    check(strict.has_value() == (std::string_view{value} == "false"),
          "strict output rejects open and schema-valued additional properties");
  }
  for (const auto value : {"null", "42", R"("false")", "[]"}) {
    const auto json = std::string{R"({"type":"object","additionalProperties":)"} + value + "}";
    check(!cail::schema_from_json(json), "additionalProperties rejects invalid value types");
  }
  const auto nested = cail::schema_from_json(
      R"({"type":"object","properties":{"headers":{"type":"object","additionalProperties":{"type":"string"}}}})");
  check(nested.has_value(), "MCP tool schemas accept nested maps");
}

void test_field_value_api() {
  cail::Field<double> confidence{
      .value = 0.5,
      .description = "Confidence score",
      .minimum = 0.0,
      .maximum = 1.0,
  };
  check(confidence.value == 0.5, "field runtime value is read through .value");
  confidence = 0.8;
  check(confidence.value == 0.8, "field assignment updates its runtime value");
  check(confidence.description == "Confidence score" && confidence.minimum == 0.0 &&
            confidence.maximum == 1.0,
        "field assignment preserves schema metadata");
}

void test_optional_schema_and_json() {
  const auto output_schema = cail::schema<Profile>();
  check(output_schema.properties.has_value(), "schema exposes model properties");
  check(output_schema.required.has_value(), "schema exposes required properties");
  if (!output_schema.properties || !output_schema.required) {
    return;
  }

  check(std::ranges::find(*output_schema.required, "answer") != output_schema.required->end(),
        "non-optional property remains required");
  check(std::ranges::find(*output_schema.required, "confidence") == output_schema.required->end(),
        "optional property stays optional in the generic schema");
  const auto& confidence = *output_schema.properties->at("confidence");
  const auto* confidence_types = std::get_if<std::vector<cail::SchemaType>>(&confidence.type);
  check(confidence_types != nullptr, "optional property schema uses a type union");
  if (confidence_types != nullptr) {
    check(*confidence_types ==
              std::vector<cail::SchemaType>{cail::SchemaType::integer, cail::SchemaType::null},
          "optional integer schema allows null");
  }

  auto missing = cail::from_json<Profile>(R"({"answer":"ok"})");
  check(missing.has_value(), "missing optional properties deserialize");
  if (missing) {
    check(!missing->confidence && !missing->address,
          "missing optional values become empty optionals");
  }

  auto null_value = cail::from_json<Profile>(R"({"answer":"ok","confidence":null,"address":null})");
  check(null_value.has_value(), "explicit null values deserialize");
  if (null_value) {
    check(!null_value->confidence && !null_value->address,
          "explicit null values become empty optionals");
  }

  Profile populated{
      .answer = "done",
      .confidence = 7,
      .address = Address{.city = "London"},
  };
  auto json = cail::to_json(populated);
  check(json && json->find(R"("confidence":7)") != std::string::npos,
        "populated optional values serialize normally");
}

void test_tool_loop() {
  std::vector<std::size_t> observed_steps;
  ScriptedClient client({
      cail::GenerationResponse{
          .tool_calls = {cail::ToolCall{
              .id = "call-1", .name = "count", .arguments = R"({"query":"abc"})"}},
          .continuation_token = "turn-1",
      },
      cail::GenerationResponse{.text = "counted"},
  });
  auto count_tool = cail::tool<ToolInput, ToolOutput>(
      "count", "Count characters", [](const ToolInput& input, const cail::ToolContext& context) {
        check(context.call_id == "call-1" && context.round == 0, "tool receives call context");
        return ToolOutput{.count = static_cast<int>(input.query.size())};
      });

  const auto result = cail::run_tool_loop(
      client,
      cail::GenerationRequest{
          .messages = {cail::Message{.content = {cail::TextPart{.text = "count abc"}}}},
          .session_id = "stable-session",
          .middleware = {cail::GenerationMiddleware{
              .after_step =
                  [&](const cail::Result<cail::GenerationResponse>& response,
                      const cail::MiddlewareContext& context) {
                    check(response.has_value(), "step middleware observes successful results");
                    observed_steps.push_back(context.step);
                  },
          }},
      },
      std::vector<cail::Tool>{count_tool});
  check(result && result->text == "counted", "tool loop returns the final model response");
  check(result && result->tool_results.size() == 1, "tool loop records tool output");
  check(result && result->tool_results.front().output == R"({"count":3})",
        "tool output stays typed and serializes");
  check(client.requests.size() == 2, "tool loop makes a follow-up model call");
  check(client.requests.size() == 2 && client.requests[1].continuation_token == "turn-1",
        "tool loop carries continuation state");
  check(client.requests.size() == 2 && client.requests[1].session_id == "stable-session",
        "tool loop carries the caller session ID");
  check(client.requests.size() == 2 && client.requests[1].messages.front().tool_call_id == "call-1",
        "tool loop sends the result for the matching call");
  check(observed_steps == std::vector<std::size_t>{0, 1},
        "middleware observes every model step in a tool loop");
}

void test_agent() {
  auto client = std::make_shared<ScriptedClient>(
      std::vector<cail::GenerationResponse>{cail::GenerationResponse{.text = "hello"}});
  cail::Agent agent({
      .model = cail::LanguageModel(
          [client](const cail::GenerationRequest& request, std::stop_token stop) {
            return client->generate(request, std::move(stop));
          }),
      .instructions = "Be concise.",
      .tools = {cail::tool<ToolInput, ToolOutput>("count", "Count characters",
                                                  [](const ToolInput& input) {
                                                    return ToolOutput{.count = static_cast<int>(
                                                                          input.query.size())};
                                                  })},
  });

  const auto response = agent.generate("Say hello.");
  check(response && response->text == "hello", "agent returns the model response");
  check(client->requests.size() == 1 && client->requests.front().messages.size() == 2,
        "agent combines its instructions with the prompt");
  check(client->requests.front().messages.front().role == cail::MessageRole::system &&
            std::get<cail::TextPart>(client->requests.front().messages.front().content.front())
                    .text == "Be concise.",
        "agent sends its instructions as the first system message");
  check(client->requests.front().tools.size() == 1 &&
            client->requests.front().tools.front().name == "count",
        "agent offers its tools on every run");
}

void test_streaming_tool_loop_and_stop_conditions() {
  ScriptedClient client({
      cail::GenerationResponse{
          .text = "checking",
          .tool_calls = {cail::ToolCall{
              .id = "call-1", .name = "count", .arguments = R"({"query":"abc"})"}},
          .provider_options = {{"state", "kept"}},
      },
      cail::GenerationResponse{.text = "counted"},
  });
  auto count_tool =
      cail::tool<ToolInput, ToolOutput>("count", "Count characters", [](const ToolInput& input) {
        return ToolOutput{.count = static_cast<int>(input.query.size())};
      });
  std::string streamed;
  const auto response =
      cail::stream_tool_loop(client, cail::GenerationRequest{}, std::vector<cail::Tool>{count_tool},
                             [&](const cail::StreamEvent& event) {
                               if (const auto* delta = std::get_if<cail::TextDelta>(&event)) {
                                 streamed += delta->text;
                               }
                             });
  check(response && response->text == "counted" && streamed == "checkingcounted",
        "streaming tool loop forwards events from every model step");
  check(response && response->tool_results.size() == 1,
        "streaming tool loop executes tools and returns their results");
  check(client.requests.size() == 2 && client.requests[1].messages.size() == 2 &&
            client.requests[1].messages.front().provider_options.contains("state"),
        "tool loop preserves provider metadata on assistant history");

  ScriptedClient stopped_client({cail::GenerationResponse{
      .tool_calls = {cail::ToolCall{
          .id = "call-1", .name = "count", .arguments = R"({"query":"abc"})"}},
  }});
  const auto stopped = cail::run_tool_loop(
      stopped_client, {}, std::vector<cail::Tool>{count_tool},
      {.stop_when = [](const cail::GenerationResponse&, const cail::MiddlewareContext& context) {
        return context.step == 0;
      }});
  check(stopped && stopped->tool_calls.size() == 1 && stopped_client.requests.size() == 1,
        "custom stop condition returns before another tool-loop step");

  std::stop_source cancelled;
  cancelled.request_stop();
  const auto cancelled_response =
      cail::stream_tool_loop(stopped_client, {}, std::vector<cail::Tool>{count_tool},
                             [](const cail::StreamEvent&) {}, {.stop = cancelled.get_token()});
  check(!cancelled_response && cancelled_response.error().code == cail::ErrorCode::cancelled,
        "streaming tool loop honors cancellation before the first step");
}

void test_text_generation_options() {
  for (bool streaming : {false, true}) {
    auto client = std::make_shared<ScriptedClient>(std::vector<cail::GenerationResponse>{
        cail::GenerationResponse{
            .text = "checking",
            .tool_calls = {cail::ToolCall{
                .id = "call-1", .name = "count", .arguments = R"({"query":"abc"})"}},
        },
        cail::GenerationResponse{.text = "counted"},
    });
    std::stop_source stop;
    auto model = cail::LanguageModel(
        [&, client](const cail::GenerationRequest& request, const std::stop_token& token) {
          check(token == stop.get_token(), "generate_text forwards cancellation token");
          return client->generate(request, token);
        },
        [&, client](const cail::GenerationRequest& request, const cail::StreamHandler& handler,
                    const std::stop_token& token) {
          check(token == stop.get_token(), "stream_text forwards cancellation token");
          return client->stream(request, handler, token);
        });
    std::vector<std::size_t> steps;
    cail::GenerateTextOptions options{
        .model = model,
        .system = "Be concise.",
        .prompt = "count abc",
        .messages = {cail::Message{
            .content = {cail::ImagePart{.bytes = "image bytes", .mime_type = "image/png"}}}},
        .tools = {cail::tool<ToolInput, ToolOutput>("count", "Count characters",
                                                    [](const ToolInput& input) {
                                                      return ToolOutput{.count = static_cast<int>(
                                                                            input.query.size())};
                                                    })},
        .structured_output = cail::StructuredOutput{.schema = cail::schema<Address>()},
        .tool_loop = {.stop = stop.get_token()},
        .session_id = "session",
        .max_output_tokens = 123,
        .stream_usage = false,
        .provider_options = {{"custom", "value"}},
        .middleware = {cail::GenerationMiddleware{
            .after_step =
                [&](const cail::Result<cail::GenerationResponse>& response,
                    const cail::MiddlewareContext& context) {
                  check(response.has_value(), "text middleware receives the result");
                  steps.push_back(context.step);
                }}},
    };
    std::string text;
    const auto result =
        streaming
            ? cail::stream_text(options,
                                [&](const cail::StreamEvent& event) {
                                  if (const auto* delta = std::get_if<cail::TextDelta>(&event)) {
                                    text += delta->text;
                                  }
                                })
            : cail::generate_text(options);
    check(result && result->text == "counted" && result->tool_results.size() == 1 &&
              result->tool_results.front().output == R"({"count":3})",
          "both text APIs execute tools and return the final result");
    check(!streaming || text == "checkingcounted", "stream_text forwards every model step");
    check(steps == std::vector<std::size_t>{0, 1}, "both text APIs run step middleware");
    check(client->requests.size() == 2, "both text APIs make a tool follow-up request");
    if (client->requests.size() != 2) {
      continue;
    }
    const auto& first = client->requests.front();
    check(first.messages.size() == 3 && first.messages.front().role == cail::MessageRole::system &&
              std::get<cail::TextPart>(first.messages.front().content.front()).text ==
                  "Be concise." &&
              std::holds_alternative<cail::ImagePart>(first.messages[1].content.front()) &&
              std::get<cail::TextPart>(first.messages.back().content.front()).text == "count abc",
          "both text APIs prepend system, preserve history, and append prompt");
    for (const auto& request : client->requests) {
      check(request.session_id == "session" && request.max_output_tokens == 123 &&
                request.stream_usage == false && request.structured_output &&
                request.provider_options.contains("custom") && request.tools.size() == 1,
            "both text APIs preserve request options through tool follow-ups");
    }
  }
}

void test_stream_text_validation_and_cancellation() {
  int calls = 0;
  auto model = cail::LanguageModel(
      [&](const cail::GenerationRequest&, const std::stop_token&) {
        ++calls;
        return cail::GenerationResponse{.text = "done"};
      },
      [&](const cail::GenerationRequest&, const cail::StreamHandler& handler,
          const std::stop_token&) {
        ++calls;
        handler(cail::ReasoningDelta{.text = "thinking"});
        handler(cail::TextDelta{.text = "done"});
        return cail::GenerationResponse{.text = "done"};
      });
  const auto ignore = [](const cail::StreamEvent&) {};
  const auto missing = cail::stream_text({.model = model, .system = "system only"}, ignore);
  check(!missing && missing.error().code == cail::ErrorCode::invalid_configuration,
        "stream_text requires prompt or messages");
  const auto missing_generate = cail::generate_text({.model = model, .system = "system only"});
  check(!missing_generate &&
            missing_generate.error().code == cail::ErrorCode::invalid_configuration,
        "generate_text shares input validation");
  const auto no_handler = cail::stream_text({.model = model, .prompt = "hello"}, {});
  check(!no_handler && no_handler.error().code == cail::ErrorCode::invalid_configuration,
        "stream_text rejects an empty handler");
  std::stop_source stop;
  stop.request_stop();
  cail::GenerateTextOptions options{
      .model = model, .prompt = "hello", .tool_loop = {.stop = stop.get_token()}};
  const auto cancelled_stream = cail::stream_text(options, ignore);
  const auto cancelled_generate = cail::generate_text(options);
  check(!cancelled_stream && cancelled_stream.error().code == cail::ErrorCode::cancelled &&
            !cancelled_generate && cancelled_generate.error().code == cail::ErrorCode::cancelled &&
            calls == 0,
        "both text APIs stop before contacting the model when cancelled");
  std::stop_source during_stream;
  std::size_t events = 0;
  const auto cancelled = cail::stream_text(
      {.model = model, .prompt = "hello", .tool_loop = {.stop = during_stream.get_token()}},
      [&](const cail::StreamEvent&) {
        ++events;
        during_stream.request_stop();
      });
  check(!cancelled && cancelled.error().code == cail::ErrorCode::cancelled && events == 2,
        "stream_text forwards typed events and observes cancellation from the callback");
  const auto unsupported = cail::stream_text(
      {.model = cail::LanguageModel([](const cail::GenerationRequest&, const std::stop_token&) {
         return cail::GenerationResponse{};
       }),
       .prompt = "hello"},
      ignore);
  check(!unsupported && unsupported.error().code == cail::ErrorCode::invalid_configuration,
        "stream_text reports models without streaming support");
}

void test_transport_retries() {
  class FlakyTransport final : public cail::HttpTransport {
  public:
    [[nodiscard]] cail::Result<cail::HttpResponse> send(const cail::HttpRequest&,
                                                        std::stop_token) override {
      ++attempts;
      return cail::HttpResponse{.status_code = attempts < 3 ? 429 : 200};
    }

    [[nodiscard]] cail::Result<cail::HttpResponse> stream(const cail::HttpRequest&,
                                                          const cail::HttpDataHandler& on_data,
                                                          std::stop_token) override {
      ++attempts;
      if (attempts == 1) {
        return cail::HttpResponse{.status_code = 503};
      }
      on_data("done");
      return cail::HttpResponse{.status_code = 200};
    }

    int attempts{};
  };

  auto send_transport = std::make_unique<FlakyTransport>();
  auto* send_flaky = send_transport.get();
  cail::RetryingHttpTransport retrying_send(std::move(send_transport),
                                            {.max_retries = 2, .initial_delay = {}});
  const auto response = retrying_send.send({}, {});
  check(response && response->status_code == 200 && send_flaky->attempts == 3,
        "transport retries 429 responses up to the configured limit");

  auto stream_transport = std::make_unique<FlakyTransport>();
  auto* stream_flaky = stream_transport.get();
  cail::RetryingHttpTransport retrying_stream(std::move(stream_transport),
                                              {.max_retries = 1, .initial_delay = {}});
  std::string data;
  const auto stream_response =
      retrying_stream.stream({}, [&](std::string_view chunk) { data.append(chunk); }, {});
  check(stream_response && data == "done" && stream_flaky->attempts == 2,
        "transport retries a 5xx stream before delivering data");

  class RetryWaitTransport final : public cail::HttpTransport {
  public:
    [[nodiscard]] cail::Result<cail::HttpResponse> send(const cail::HttpRequest&,
                                                        std::stop_token) override {
      sent.set_value();
      return cail::HttpResponse{.status_code = 503};
    }
    [[nodiscard]] cail::Result<cail::HttpResponse>
    stream(const cail::HttpRequest&, const cail::HttpDataHandler&, std::stop_token) override {
      return std::unexpected(cail::Error{.code = cail::ErrorCode::invalid_configuration});
    }
    std::promise<void> sent;
  };
  auto retry_wait_transport = std::make_unique<RetryWaitTransport>();
  auto sent = retry_wait_transport->sent.get_future();
  cail::RetryingHttpTransport retry_wait(
      std::move(retry_wait_transport),
      {.max_retries = 2, .initial_delay = std::chrono::seconds{5}});
  std::stop_source stop;
  auto pending =
      std::async(std::launch::async, [&] { return retry_wait.send({}, stop.get_token()); });
  sent.wait();
  const auto began = std::chrono::steady_clock::now();
  stop.request_stop();
  const auto cancelled = pending.get();
  check(!cancelled && cancelled.error().code == cail::ErrorCode::cancelled &&
            std::chrono::steady_clock::now() - began < std::chrono::seconds{2},
        "cancellation interrupts the HTTP retry delay");

  class TimeoutTransport final : public cail::HttpTransport {
  public:
    [[nodiscard]] cail::Result<cail::HttpResponse> send(const cail::HttpRequest& request,
                                                        std::stop_token) override {
      timeout = request.timeout;
      return cail::HttpResponse{.status_code = 200};
    }
    [[nodiscard]] cail::Result<cail::HttpResponse>
    stream(const cail::HttpRequest&, const cail::HttpDataHandler&, std::stop_token) override {
      return cail::HttpResponse{.status_code = 200};
    }
    std::chrono::seconds timeout{};
  };
  auto timeout_transport = std::make_unique<TimeoutTransport>();
  auto* timeout_probe = timeout_transport.get();
  cail::TimeoutHttpTransport configured_timeout(std::move(timeout_transport),
                                                std::chrono::seconds{7});
  static_cast<void>(configured_timeout.send({}, {}));
  check(timeout_probe->timeout == std::chrono::seconds{7},
        "transport applies the configured request timeout");
}

void test_async_transport_retries() {
  class AsyncTransport final : public cail::HttpTransport {
  public:
    [[nodiscard]] cail::Result<cail::HttpResponse> send(const cail::HttpRequest&,
                                                        std::stop_token) override {
      return std::unexpected(cail::Error{.code = cail::ErrorCode::invalid_configuration});
    }
    void send_async(cail::HttpRequest, cail::HttpCompletion complete, std::stop_token) override {
      ++attempts;
      complete(cail::HttpResponse{.status_code = attempts < 3 ? 503 : 200});
    }
    [[nodiscard]] cail::Result<cail::HttpResponse>
    stream(const cail::HttpRequest&, const cail::HttpDataHandler&, std::stop_token) override {
      return std::unexpected(cail::Error{.code = cail::ErrorCode::invalid_configuration});
    }
    int attempts{};
  };

  auto transport = std::make_unique<AsyncTransport>();
  auto* probe = transport.get();
  cail::RetryingHttpTransport retrying(std::move(transport),
                                       {.max_retries = 2, .initial_delay = {}});
  std::promise<cail::Result<cail::HttpResponse>> completed;
  auto result = completed.get_future();
  retrying.send_async(
      {},
      [&](cail::Result<cail::HttpResponse> response) { completed.set_value(std::move(response)); },
      {});
  const auto response = result.get();
  check(response && response->status_code == 200 && probe->attempts == 3,
        "async transport retries 5xx responses without blocking the caller");

  auto cancel_transport = std::make_unique<AsyncTransport>();
  cail::RetryingHttpTransport cancel_retry(
      std::move(cancel_transport), {.max_retries = 2, .initial_delay = std::chrono::seconds{5}});
  std::promise<cail::Result<cail::HttpResponse>> cancelled;
  auto cancelled_result = cancelled.get_future();
  std::stop_source stop;
  cancel_retry.send_async(
      {}, [&](cail::Result<cail::HttpResponse> value) { cancelled.set_value(std::move(value)); },
      stop.get_token());
  stop.request_stop();
  check(cancelled_result.wait_for(std::chrono::seconds{1}) == std::future_status::ready,
        "cancellation interrupts an async retry delay");
  const auto stopped = cancelled_result.get();
  check(!stopped && stopped.error().code == cail::ErrorCode::cancelled,
        "async retry reports cancellation");
}

void test_tool_loop_partial_progress() {
  cail::detail::ToolLoop loop({}, {}, {.max_rounds = 0});
  auto finished = loop.begin_step(cail::GenerationResponse{
      .text = "Calling a tool",
      .usage = cail::TokenUsage{.input_tokens = 3, .output_tokens = 2},
      .tool_calls = {{.id = "call", .name = "missing", .arguments = "{}"}},
  });
  check(!finished, "round limit fails");
  auto error = loop.with_progress(finished.error());
  check(error.code == cail::ErrorCode::tool_loop_limit && error.partial_response &&
            error.partial_response->steps.size() == 1 && error.partial_response->turn.size() == 1 &&
            error.partial_response->total_usage->input_tokens == 3,
        "round limit preserves model progress and usage");

  std::stop_source stop;
  cail::detail::ToolLoop cancelled({}, {}, {.stop = stop.get_token()});
  check(cancelled
            .begin_step(cail::GenerationResponse{
                .tool_calls = {{.id = "call", .name = "write", .arguments = "{}"}},
            })
            .has_value(),
        "tool step begins");
  stop.request_stop();
  auto accepted = cancelled.accept_tool(std::string{"done"});
  auto cancellation = cancelled.with_progress(accepted.error());
  check(cancellation.code == cail::ErrorCode::cancelled &&
            cancellation.partial_response->tool_results.size() == 1 &&
            cancellation.partial_response->turn.size() == 2,
        "cancellation preserves successful tool output");
  cail::detail::ToolLoop empty({}, {}, {});
  check(!empty.with_progress(cail::generation_cancelled_error()).partial_response,
        "failure before a model response has no partial progress");
}

} // namespace test

int main() {
  test::test_tool_loop_partial_progress();
  test::test_length_constraints();
  test::test_constraint_validation();
  test::test_control_character_json();
  test::test_additional_properties_schema();
  test::test_field_value_api();
  test::test_optional_schema_and_json();
  test::test_tool_loop();
  test::test_agent();
  test::test_streaming_tool_loop_and_stop_conditions();
  test::test_text_generation_options();
  test::test_stream_text_validation_and_cancellation();
  test::test_transport_retries();
  test::test_async_transport_retries();
  return test::failures == 0 ? 0 : 1;
}
