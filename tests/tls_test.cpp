#include "detail/glaze_http_transport.hpp"
#include "test_support.hpp"
#include <cail/detail/system_ca.hpp>
#include <glaze/net/http_client.hpp>

#include <cstdlib>
#include <string>
#include <system_error>

int main(int argc, char** argv) {
  if (argc != 3) {
    return 1;
  }
  const std::string mode = argv[2];
  try {
    cail::detail::GlazeHttpTransport transport;
    const auto response =
        transport.send({.method = "GET", .url = argv[1], .timeout = std::chrono::seconds(15)}, {});
    test::check(mode != "invalid", "invalid CA file must report a configuration error");
    test::check(response.has_value() == (mode == "trusted"),
                "TLS verifies the configured trust roots and hostname" +
                    (response ? std::string{} : ": " + response.error().message));
    if (response) {
      test::check(response->status_code == 200, "trusted HTTPS reaches the server");
    } else if (mode == "untrusted" || mode == "system") {
      test::check(response.error().message.find("certificate verify failed") != std::string::npos,
                  "untrusted certificates fail verification rather than timing out");
    }
  } catch (const std::system_error&) {
    test::check(mode == "invalid", "valid CA configuration must load successfully");
  }
  if (mode == "system" || (mode == "trusted" && std::getenv("SSL_CERT_FILE") != nullptr)) {
    glz::http_client client;
    test::check(cail::detail::configure_system_ca(client).has_value(), "system trust loads");
    client.configure_ssl_context([&](auto& context) {
      const int count = sk_X509_OBJECT_num(
          X509_STORE_get0_objects(SSL_CTX_get_cert_store(context.native_handle())));
      test::check(mode == "system" ? count > 0 : count == 1,
                  "system roots load and an explicit CA file replaces them");
    });
  }
  return test::failures == 0 ? 0 : 1;
}
