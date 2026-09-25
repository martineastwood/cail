#include <cail/cail.hpp>

#include <optional>
#include <string>

struct Response {
    cail::Field<double> confidence{.description = "Confidence"};
    std::optional<std::string> explanation;
};

int main()
{
    Response response{.confidence = {.value = 0.8}, .explanation = "clear"};
    const auto json = cail::to_json(response);
    if (!json) {
        return 1;
    }
    const auto decoded = cail::from_json<Response>(*json);
    const auto model = cail::openrouter("openai/gpt-4o-mini");
    const auto anthropic = cail::anthropic("claude-haiku-4-5-20251001");
    const auto gemini = cail::gemini("gemini-3.5-flash-lite");
    return decoded && decoded->confidence.value == 0.8 && decoded->explanation == "clear" &&
                   model.capabilities().structured_output && anthropic.capabilities().tools &&
                   gemini.capabilities().streaming ? 0 : 1;
}
