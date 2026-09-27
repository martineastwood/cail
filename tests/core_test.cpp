#include "test_support.hpp"

#include <cail/field.hpp>
#include <cail/json.hpp>
#include <cail/schema.hpp>
#include <cail/tool.hpp>

#include <algorithm>

namespace test {

void test_control_character_json() {
  std::string controls;
  for (int ch = 0; ch < 32; ++ch) controls += static_cast<char>(ch);
  const auto encoded = cail::to_json(Address{.city = controls});
  check(encoded.has_value(), "control characters serialize");
  if (!encoded) return;
  const auto decoded = cail::from_json<Address>(*encoded);
  check(decoded && decoded->city == controls,
        "JSON preserves every control character");
  const auto merged = cail::merge_json_objects(*encoded, R"({"extra":"\u0000"})");
  check(merged.has_value(), "provider options merge with control characters");
  if (!merged) return;
  glz::generic parsed;
  const auto error = glz::read_json(parsed, *merged);
  check(!error && parsed["city"].get<std::string>() == controls &&
            parsed["extra"].get<std::string>() == std::string(1, '\0'),
        "merged JSON preserves control characters in both objects");
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
  check(confidence.description == "Confidence score" &&
            confidence.minimum == 0.0 && confidence.maximum == 1.0,
        "field assignment preserves schema metadata");
}

void test_optional_schema_and_json() {
  const auto output_schema = cail::schema<Profile>();
  check(output_schema.properties.has_value(),
        "schema exposes model properties");
  check(output_schema.required.has_value(),
        "schema exposes required properties");
  if (!output_schema.properties || !output_schema.required) {
    return;
  }

  check(std::ranges::find(*output_schema.required, "answer") !=
            output_schema.required->end(),
        "non-optional property remains required");
  check(std::ranges::find(*output_schema.required, "confidence") ==
            output_schema.required->end(),
        "optional property stays optional in the generic schema");
  const auto &confidence = *output_schema.properties->at("confidence");
  const auto *confidence_types =
      std::get_if<std::vector<cail::SchemaType>>(&confidence.type);
  check(confidence_types != nullptr,
        "optional property schema uses a type union");
  if (confidence_types != nullptr) {
    check(*confidence_types ==
              std::vector<cail::SchemaType>{cail::SchemaType::integer,
                                            cail::SchemaType::null},
          "optional integer schema allows null");
  }

  auto missing = cail::from_json<Profile>(R"({"answer":"ok"})");
  check(missing.has_value(), "missing optional properties deserialize");
  if (missing) {
    check(!missing->confidence && !missing->address,
          "missing optional values become empty optionals");
  }

  auto null_value = cail::from_json<Profile>(
      R"({"answer":"ok","confidence":null,"address":null})");
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
  ScriptedClient client({
      cail::GenerationResponse{
          .tool_calls = {cail::ToolCall{.id = "call-1",
                                        .name = "count",
                                        .arguments = R"({"query":"abc"})"}},
          .continuation_token = "turn-1",
      },
      cail::GenerationResponse{.text = "counted"},
  });
  auto count_tool = cail::tool<ToolInput, ToolOutput>(
      "count", "Count characters",
      [](const ToolInput &input, const cail::ToolContext &context) {
        check(context.call_id == "call-1" && context.round == 0,
              "tool receives call context");
        return ToolOutput{.count = static_cast<int>(input.query.size())};
      });

  const auto result = cail::run_tool_loop(
      client,
      cail::GenerationRequest{
          .messages = {cail::Message{
              .content = {cail::TextPart{.text = "count abc"}}}},
          .session_id = "stable-session",
      },
      std::vector<cail::Tool>{count_tool});
  check(result && result->text == "counted",
        "tool loop returns the final model response");
  check(result && result->tool_results.size() == 1,
        "tool loop records tool output");
  check(result && result->tool_results.front().output == R"({"count":3})",
        "tool output stays typed and serializes");
  check(client.requests.size() == 2, "tool loop makes a follow-up model call");
  check(client.requests.size() == 2 &&
            client.requests[1].continuation_token == "turn-1",
        "tool loop carries continuation state");
  check(client.requests.size() == 2 &&
            client.requests[1].session_id == "stable-session",
        "tool loop carries the caller session ID");
  check(client.requests.size() == 2 &&
            client.requests[1].messages.front().tool_call_id == "call-1",
        "tool loop sends the result for the matching call");
}

} // namespace test

int main() {
  test::test_control_character_json();
  test::test_field_value_api();
  test::test_optional_schema_and_json();
  test::test_tool_loop();
  return test::failures == 0 ? 0 : 1;
}
