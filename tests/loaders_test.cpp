#include "test_support.hpp"

#include <cail/anthropic.hpp>
#include <cail/chat_completions.hpp>
#include <cail/gemini.hpp>
#include <cail/loaders.hpp>
#include <cail/openai.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>

int main() {
  using test::check;
  const auto directory =
      std::filesystem::temp_directory_path() /
      ("cail-loaders-" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directory(directory);
  const auto write = [&](std::string_view name, std::string_view bytes) {
    std::ofstream file(directory / name, std::ios::binary);
    file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  };
  write("text.txt", "hello\r\nworld");
  const auto text = cail::load_text(directory / "text.txt");
  check(text && text->text == "hello\r\nworld", "text preserves line endings");
  write("empty.txt", "");
  check(cail::load_text(directory / "empty.txt").has_value(), "empty text loads");
  check(!cail::load_file(directory / "missing"), "missing files fail");
  check(!cail::load_file(directory), "directories fail");
  check(!cail::load_file(directory / "text.txt", 4), "oversized files fail");
  check(!cail::load_image(directory / "text.txt"), "non-images fail");
  check(!cail::load_pdf(directory / "text.txt"), "non-PDFs fail");
  const std::string png{"\x89PNG\r\n\x1a\n\0", 9};
  write("picture.dat", png);
  const auto image = cail::load_image(directory / "picture.dat");
  check(image && image->bytes == png && image->mime_type == "image/png",
        "image detection uses bytes and preserves NULs");
  for (const auto& [bytes, mime] :
       std::vector<std::pair<std::string, std::string>>{{"\xff\xd8\xff", "image/jpeg"},
                                                        {"GIF89a", "image/gif"},
                                                        {"RIFFxxxxWEBP", "image/webp"}}) {
    write("image", bytes);
    const auto loaded = cail::load_image(directory / "image");
    check(loaded && loaded->mime_type == mime, "image MIME type is detected");
  }
  write("report.pdf", "%PDF-1.7\n");
  const auto pdf = cail::load_pdf(directory / "report.pdf");
  check(pdf && pdf->filename == "report.pdf" && pdf->bytes == "%PDF-1.7\n",
        "PDF bytes and filename load");
  if (pdf) {
    cail::GenerationRequest request{
        .messages = {cail::Message{.content = {cail::TextPart{.text = "Summarize"}, *pdf}}}};
    const auto saved = cail::to_json(request.messages.front());
    const auto restored =
        saved ? cail::from_json<cail::Message>(*saved) : cail::Result<cail::Message>{};
    check(restored && std::holds_alternative<cail::PdfPart>(restored->content.back()) &&
              std::get<cail::PdfPart>(restored->content.back()).bytes == pdf->bytes,
          "PDF content survives conversation JSON serialization");
    auto transport = std::make_unique<test::StubTransport>();
    auto* stub = transport.get();
    auto model = cail::create_openai({.api_key = "test"})("test", std::move(transport));
    check(model.adapter_capabilities().pdf_input, "OpenAI reports PDF input support");
    check(model.generate(request).has_value() &&
              stub->request.body.find("input_file") != std::string::npos,
          "OpenAI sends PDF input through the generation API");
    const auto openai = cail::detail::openai::wire::input_content(request.messages.front());
    check(openai && openai->str.find("input_file") != std::string::npos &&
              openai->str.find("data:application/pdf;base64,JVBERi0xLjcK") != std::string::npos,
          "OpenAI encodes a PDF data URL");
    const auto anthropic = cail::detail::anthropic::encode(request, {.model = "test"}, false);
    const auto anthropic_json = anthropic ? cail::to_json(*anthropic) : cail::Result<std::string>{};
    check(anthropic && anthropic_json && anthropic_json->find("document") != std::string::npos &&
              anthropic_json->find("JVBERi0xLjcK") != std::string::npos,
          "Anthropic encodes a base64 document");
    const auto gemini = cail::detail::gemini::encode(request);
    const auto gemini_json = gemini ? cail::to_json(*gemini) : cail::Result<std::string>{};
    check(gemini && gemini_json && gemini_json->find("application/pdf") != std::string::npos &&
              gemini_json->find("JVBERi0xLjcK") != std::string::npos,
          "Gemini encodes an inline PDF");
    check(!cail::detail::chat_completions::encode(request, "test", false, false),
          "Chat Completions rejects PDFs");
    request.messages.front().role = cail::MessageRole::tool;
    check(!cail::detail::validate_pdf_parts(request), "PDFs in tool results fail");
    request.messages.front().role = cail::MessageRole::assistant;
    check(!cail::detail::validate_pdf_parts(request), "PDFs in assistant messages fail");
    request.messages.front().role = cail::MessageRole::user;
    std::get<cail::PdfPart>(request.messages.front().content.back()).bytes.clear();
    check(!cail::detail::validate_pdf_parts(request), "empty PDF parts fail");
  }
  std::filesystem::remove_all(directory);
  return test::failures == 0 ? 0 : 1;
}
