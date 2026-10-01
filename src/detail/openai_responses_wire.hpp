#pragma once

#include <cail/generation.hpp>
#include <glaze/core/common.hpp>

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace cail::detail::openai::wire {

struct InputMessage {
  std::string role;
  glz::raw_json content;
};

struct InputTextPart {
  std::string type{"input_text"};
  std::string text;
};

struct InputImagePart {
  std::string type{"input_image"};
  std::string image_url;
};

struct InputFilePart {
  std::string type{"input_file"};
  std::string filename;
  std::string file_data;
};

struct FunctionCallInput {
  std::string type{"function_call"};
  std::string call_id;
  std::string name;
  std::string arguments;
};

struct FunctionCallOutputInput {
  std::string type{"function_call_output"};
  std::string call_id;
  std::string output;
};

struct FunctionTool {
  std::string type{"function"};
  std::string name;
  std::string description;
  glz::raw_json parameters;
  bool strict{true};
};

struct JsonSchemaFormat {
  std::string type{"json_schema"};
  std::string name;
  std::optional<std::string> description;
  bool strict{true};
  glz::raw_json schema;
};

struct TextOptions {
  JsonSchemaFormat format;
};

struct RequestBody {
  std::string model;
  std::vector<glz::raw_json> input;
  std::optional<std::vector<FunctionTool>> tools;
  std::optional<std::string> previous_response_id;
  std::optional<TextOptions> text;
  std::optional<bool> stream;
  std::optional<std::size_t> max_output_tokens;
  std::optional<double> temperature;
  std::optional<double> top_p;
  std::optional<glz::generic> tool_choice;
};

struct ResponseContent {
  std::string type;
  std::optional<std::string> text;
  std::optional<std::string> refusal;
};

struct ResponseItem {
  std::string type;
  std::vector<ResponseContent> content;
  std::vector<ResponseContent> summary;
  std::optional<std::string> call_id;
  std::optional<std::string> name;
  std::optional<std::string> arguments;
  std::optional<std::string> encrypted_content;
};

struct ResponseUsage {
  std::size_t input_tokens{};
  std::size_t output_tokens{};
  struct InputDetails {
    std::optional<std::size_t> cached_tokens;
  };
  struct OutputDetails {
    std::optional<std::size_t> reasoning_tokens;
  };
  std::optional<InputDetails> input_tokens_details;
  std::optional<OutputDetails> output_tokens_details;
};

struct ProviderError {
  std::string message;
  std::optional<std::string> code;
  std::optional<std::string> type;
};

struct ResponseBody {
  std::optional<std::string> id;
  std::string status;
  std::vector<glz::raw_json> output;
  std::optional<ResponseUsage> usage;
  std::optional<ProviderError> error;
  struct IncompleteDetails {
    std::optional<std::string> reason;
  };
  std::optional<IncompleteDetails> incomplete_details;
};

struct StreamEventBody {
  std::string type;
  std::optional<std::string> delta;
  std::optional<std::size_t> output_index;
  std::optional<std::string> message;
  std::optional<ResponseBody> response;
  std::optional<ProviderError> error;
};

[[nodiscard]] Result<ResponseItem> decode_response_item(const glz::raw_json& raw, int http_status);

[[nodiscard]] Result<GenerationResponse> decode_response(ResponseBody response_body,
                                                         int http_status);

[[nodiscard]] std::string_view role_name(MessageRole role);

[[nodiscard]] Result<std::string> text_content(const Message& message);

[[nodiscard]] Result<glz::raw_json> input_content(const Message& message);

} // namespace cail::detail::openai::wire
