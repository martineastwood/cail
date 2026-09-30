#include <cail/cail.hpp>

#include <iostream>
#include <memory>
#include <utility>

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: cail_openai_multimodal_memory report.pdf chart.png\n";
    return 1;
  }
  auto pdf = cail::load_pdf(argv[1]);
  auto image = cail::load_image(argv[2]);
  if (!pdf || !image) {
    std::cerr << (!pdf ? pdf.error().message : image.error().message) << '\n';
    return 1;
  }
  cail::Agent agent({
      .model = cail::openai("gpt-6-luna"),
      .instructions = "You help readers understand reports and charts.",
      .memory = std::make_shared<cail::FileConversationMemory>("conversations"),
      .conversation_id = "report-review",
  });
  auto first = agent.generate(cail::Message{
      .content = {cail::TextPart{.text = "Compare this chart with the report's findings."},
                  std::move(*image), std::move(*pdf)},
  });
  if (!first) {
    std::cerr << first.error().message << '\n';
    return 1;
  }
  std::cout << first->text << "\n\n";
  auto follow_up = agent.generate("Which finding best explains the chart?");
  if (!follow_up) {
    std::cerr << follow_up.error().message << '\n';
    return 1;
  }
  std::cout << follow_up->text << '\n';
}
