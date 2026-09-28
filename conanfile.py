import os
import re

from conan import ConanFile
from conan.tools.build import check_min_cppstd
from conan.tools.files import copy, load

required_conan_version = ">=2.0"


class CailConan(ConanFile):
    name = "cail"
    description = (
        "A typed C++ SDK for LLM providers, with a provider-neutral model and generation API"
    )
    license = "MIT"
    url = "https://github.com/martineastwood/cail"
    homepage = "https://github.com/martineastwood/cail"
    topics = ("llm", "ai", "openai", "anthropic", "gemini", "streaming", "tools", "header-only")
    package_type = "header-library"
    settings = "os", "arch", "compiler", "build_type"
    exports_sources = "include/*", "LICENSE"
    no_copy_source = True

    def set_version(self):
        # Keep one source of truth for the version.
        if self.version:
            return
        content = load(self, os.path.join(self.recipe_folder, "CMakeLists.txt"))
        self.version = re.search(r"project\(\s*cail\s+VERSION\s+([0-9.]+)", content).group(1)

    def requirements(self):
        # CAIL's public headers include all of these, so each one is transitive.
        self.requires("glaze/8.4.0", transitive_headers=True)
        self.requires("magic_enum/0.9.8", transitive_headers=True)
        # The Conan Center glaze package is header-only and ships no CMake module,
        # so the Asio backend is named here rather than selected by Glaze.
        self.requires("asio/1.38.2", transitive_headers=True)
        self.requires("openssl/[>=3.0 <4]", transitive_headers=True, transitive_libs=True)

    def validate(self):
        check_min_cppstd(self, 23)

    def package_id(self):
        self.info.clear()

    def package(self):
        copy(
            self,
            "*.hpp",
            src=os.path.join(self.source_folder, "include"),
            dst=os.path.join(self.package_folder, "include"),
            keep_path=True,
        )
        copy(
            self,
            "LICENSE",
            src=self.source_folder,
            dst=os.path.join(self.package_folder, "licenses"),
        )

    def package_info(self):
        self.cpp_info.bindirs = []
        self.cpp_info.libdirs = []
        # Glaze gates HTTPS behind this define, and its Conan package does not set it.
        # CAIL's providers all use HTTPS.
        self.cpp_info.defines = ["GLZ_ENABLE_SSL"]
        self.cpp_info.requires = [
            "glaze::glaze",
            "magic_enum::magic_enum",
            "asio::asio",
            "openssl::ssl",
            "openssl::crypto",
        ]
