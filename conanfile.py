import os
import re

from conan import ConanFile
from conan.tools.build import check_min_cppstd
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
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
    topics = ("llm", "ai", "openai", "anthropic", "gemini", "streaming", "tools")
    package_type = "static-library"
    settings = "os", "arch", "compiler", "build_type"
    exports_sources = "CMakeLists.txt", "cmake/*", "include/*", "src/*", "LICENSE"

    def set_version(self):
        if self.version:
            return
        content = load(self, os.path.join(self.recipe_folder, "CMakeLists.txt"))
        self.version = re.search(r"project\(\s*cail\s+VERSION\s+([0-9.]+)", content).group(1)

    def requirements(self):
        self.requires("glaze/8.4.0", transitive_headers=True)
        self.requires("magic_enum/0.9.8", transitive_headers=True)
        # The coroutine API exposes Asio; HTTPS needs OpenSSL at link time.
        self.requires("asio/1.38.2", transitive_headers=True)
        self.requires("openssl/[>=3.0 <4]", transitive_libs=True)

    def validate(self):
        check_min_cppstd(self, 23)

    def layout(self):
        cmake_layout(self)

    def generate(self):
        CMakeDeps(self).generate()
        toolchain = CMakeToolchain(self)
        toolchain.variables["CAIL_BUILD_EXAMPLES"] = False
        toolchain.variables["BUILD_TESTING"] = False
        toolchain.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        CMake(self).install()
        copy(
            self,
            "LICENSE",
            src=self.source_folder,
            dst=os.path.join(self.package_folder, "licenses"),
        )

    def package_info(self):
        self.cpp_info.libs = ["cail"]
        if self.settings.os == "Macos":
            self.cpp_info.cxxflags = ["-fexperimental-library"]
        self.cpp_info.requires = [
            "glaze::glaze",
            "magic_enum::magic_enum",
            "asio::asio",
            "openssl::ssl",
            "openssl::crypto",
        ]
