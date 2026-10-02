#include <cail/detail/system_ca.hpp>

#include <glaze/net/http_client.hpp>

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <type_traits>

#ifdef __APPLE__
#include <Security/Security.h>
#endif

namespace cail::detail {

std::expected<void, std::error_code> configure_system_ca(glz::http_client& client) {
  const auto* file = std::getenv("SSL_CERT_FILE");
  const auto* directory = std::getenv("SSL_CERT_DIR");
  const bool custom_file = file != nullptr && *file != '\0';
  const bool custom_directory = directory != nullptr && *directory != '\0';
  std::error_code error;
  client.configure_ssl_context([&](auto& context) {
#ifdef GLZ_USING_BOOST_ASIO
    glz::asio::error_code asio_error;
#else
    std::error_code asio_error;
#endif
    // OpenSSL takes ownership of the new store. Do not keep build-machine trust paths.
    std::unique_ptr<X509_STORE, decltype(&X509_STORE_free)> store(X509_STORE_new(),
                                                                  X509_STORE_free);
    if (!store) {
      error = std::make_error_code(std::errc::not_enough_memory);
      return;
    }
    SSL_CTX_set_cert_store(context.native_handle(), store.release());
    if (custom_file || custom_directory) {
      if (custom_file) {
        context.load_verify_file(file, asio_error);
        error = asio_error;
      }
      if (!error && custom_directory) {
        context.add_verify_path(directory, asio_error);
        error = asio_error;
      }
      return;
    }
#ifdef __APPLE__
    CFArrayRef anchors = nullptr;
    if (SecTrustCopyAnchorCertificates(&anchors) != errSecSuccess || anchors == nullptr) {
      error = std::make_error_code(std::errc::io_error);
      return;
    }
    const auto release = [](auto* value) { CFRelease(value); };
    std::unique_ptr<std::remove_pointer_t<CFArrayRef>, decltype(release)> owned_anchors(anchors,
                                                                                        release);
    for (CFIndex index = 0; index < CFArrayGetCount(anchors); ++index) {
      // CFArray returns an untyped borrowed reference; Apple's API accepts a mutable ref.
      const auto certificate =
          static_cast<SecCertificateRef>(const_cast<void*>(CFArrayGetValueAtIndex(anchors, index)));
      std::unique_ptr<std::remove_pointer_t<CFDataRef>, decltype(release)> data(
          SecCertificateCopyData(certificate), release);
      if (!data) {
        error = std::make_error_code(std::errc::io_error);
        return;
      }
      const auto* bytes = CFDataGetBytePtr(data.get());
      std::unique_ptr<X509, decltype(&X509_free)> root(
          d2i_X509(nullptr, &bytes, CFDataGetLength(data.get())), X509_free);
      if (!root ||
          X509_STORE_add_cert(SSL_CTX_get_cert_store(context.native_handle()), root.get()) != 1) {
        ERR_clear_error();
        error = std::make_error_code(std::errc::io_error);
        return;
      }
    }
#elif defined(__linux__)
    for (const auto* path :
         {"/etc/ssl/certs/ca-certificates.crt", "/etc/pki/tls/certs/ca-bundle.crt",
          "/etc/ssl/ca-bundle.pem", "/etc/pki/tls/cacert.pem",
          "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem", "/etc/ssl/cert.pem"}) {
      std::error_code exists_error;
      if (std::filesystem::is_regular_file(path, exists_error)) {
        context.load_verify_file(path, asio_error);
        error = asio_error;
        return;
      }
    }
    // Keep local HTTP usable on minimal systems without ca-certificates.
#else
    context.set_default_verify_paths(asio_error);
    error = asio_error;
    (void)glz::detail::load_os_ca_certificates(context);
#endif
  });
  if (error) {
    return std::unexpected(error);
  }
  return {};
}

} // namespace cail::detail
