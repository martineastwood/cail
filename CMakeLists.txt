cmake_minimum_required(VERSION 3.31)

project(cail VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CTest)

include(FetchContent)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# HTTPS is required by provider clients.
set(glaze_INSTALL ON)
set(glaze_ENABLE_SSL ON)
# Glaze 8.4.0 applies its SYSTEM include setting to target_link_libraries for an
# INTERFACE target when SSL is enabled. Keep that setting disabled for its CMake target.
set(glaze_INCLUDES_WITH_SYSTEM OFF)

FetchContent_Declare(
    glaze
    GIT_REPOSITORY https://github.com/stephenberry/glaze.git
    GIT_TAG v8.4.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(glaze)

include("${glaze_SOURCE_DIR}/cmake/glaze-asio.cmake")
glaze_setup_asio()

FetchContent_Declare(
    magic_enum
    GIT_REPOSITORY https://github.com/Neargye/magic_enum.git
    GIT_TAG v0.9.8
    GIT_SHALLOW TRUE
)
set(MAGIC_ENUM_OPT_INSTALL ON)
FetchContent_MakeAvailable(magic_enum)

add_library(cail INTERFACE)
add_library(cail::cail ALIAS cail)

target_compile_features(cail INTERFACE cxx_std_23)
# Glaze creates glaze::asio dynamically, so carry its build requirements here
# and reattach the target from the installed package config.
target_include_directories(cail INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    "$<BUILD_INTERFACE:$<TARGET_PROPERTY:glaze::asio,INTERFACE_INCLUDE_DIRECTORIES>>"
)
target_compile_definitions(cail INTERFACE
    "$<BUILD_INTERFACE:$<TARGET_PROPERTY:glaze::asio,INTERFACE_COMPILE_DEFINITIONS>>"
)
target_link_libraries(cail INTERFACE
    glaze::glaze
    "$<BUILD_INTERFACE:$<TARGET_PROPERTY:glaze::asio,INTERFACE_LINK_LIBRARIES>>"
    magic_enum::magic_enum
)

set(CAIL_CMAKE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/cail")
configure_package_config_file(
    cmake/cailConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/cailConfig.cmake"
    INSTALL_DESTINATION "${CAIL_CMAKE_INSTALL_DIR}"
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/cailConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
)

install(TARGETS cail EXPORT cailTargets INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(DIRECTORY include/cail DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(EXPORT cailTargets NAMESPACE cail:: DESTINATION "${CAIL_CMAKE_INSTALL_DIR}")
install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/cailConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/cailConfigVersion.cmake"
    DESTINATION "${CAIL_CMAKE_INSTALL_DIR}"
)

option(CAIL_BUILD_EXAMPLES "Build the SDK examples" ON)

if(CAIL_BUILD_EXAMPLES)
    add_executable(cail_typed_schema examples/typed_schema.cpp)
    target_link_libraries(cail_typed_schema PRIVATE cail::cail)

    add_executable(cail_openai_prompt examples/openai_prompt.cpp)
    target_link_libraries(cail_openai_prompt PRIVATE cail::cail)

    add_executable(cail_openai_stream examples/openai_stream.cpp)
    target_link_libraries(cail_openai_stream PRIVATE cail::cail)

    add_executable(cail_openai_create examples/openai_create.cpp)
    target_link_libraries(cail_openai_create PRIVATE cail::cail)

    add_executable(cail_openai_generate_object examples/openai_generate_object.cpp)
    target_link_libraries(cail_openai_generate_object PRIVATE cail::cail)

    add_executable(cail_openai_tool_call examples/openai_tool_call.cpp)
    target_link_libraries(cail_openai_tool_call PRIVATE cail::cail)

    add_executable(cail_openai_embed examples/openai_embed.cpp)
    target_link_libraries(cail_openai_embed PRIVATE cail::cail)

    add_executable(cail_openrouter_prompt examples/openrouter_prompt.cpp)
    target_link_libraries(cail_openrouter_prompt PRIVATE cail::cail)

    add_executable(cail_openrouter_stream examples/openrouter_stream.cpp)
    target_link_libraries(cail_openrouter_stream PRIVATE cail::cail)

    add_executable(cail_openrouter_object examples/openrouter_object.cpp)
    target_link_libraries(cail_openrouter_object PRIVATE cail::cail)

    add_executable(cail_opencode_prompt examples/opencode_prompt.cpp)
    target_link_libraries(cail_opencode_prompt PRIVATE cail::cail)

    add_executable(cail_anthropic_prompt examples/anthropic_prompt.cpp)
    target_link_libraries(cail_anthropic_prompt PRIVATE cail::cail)

    add_executable(cail_anthropic_stream examples/anthropic_stream.cpp)
    target_link_libraries(cail_anthropic_stream PRIVATE cail::cail)

    add_executable(cail_anthropic_object examples/anthropic_object.cpp)
    target_link_libraries(cail_anthropic_object PRIVATE cail::cail)

    add_executable(cail_anthropic_tool_call examples/anthropic_tool_call.cpp)
    target_link_libraries(cail_anthropic_tool_call PRIVATE cail::cail)

    add_executable(cail_gemini_prompt examples/gemini_prompt.cpp)
    target_link_libraries(cail_gemini_prompt PRIVATE cail::cail)

    add_executable(cail_gemini_stream examples/gemini_stream.cpp)
    target_link_libraries(cail_gemini_stream PRIVATE cail::cail)

    add_executable(cail_gemini_object examples/gemini_object.cpp)
    target_link_libraries(cail_gemini_object PRIVATE cail::cail)

    add_executable(cail_gemini_tool_call examples/gemini_tool_call.cpp)
    target_link_libraries(cail_gemini_tool_call PRIVATE cail::cail)

    add_executable(cail_mistral_prompt examples/mistral_prompt.cpp)
    target_link_libraries(cail_mistral_prompt PRIVATE cail::cail)

    add_executable(cail_hyper_prompt examples/hyper_prompt.cpp)
    target_link_libraries(cail_hyper_prompt PRIVATE cail::cail)

    add_executable(cail_ollama_cloud_prompt examples/ollama_cloud_prompt.cpp)
    target_link_libraries(cail_ollama_cloud_prompt PRIVATE cail::cail)

    add_executable(cail_local_prompt examples/local_prompt.cpp)
    target_link_libraries(cail_local_prompt PRIVATE cail::cail)

    add_executable(cail_local_object examples/local_object.cpp)
    target_link_libraries(cail_local_object PRIVATE cail::cail)

    add_executable(cail_local_tool_call examples/local_tool_call.cpp)
    target_link_libraries(cail_local_tool_call PRIVATE cail::cail)

    add_executable(cail_foundry_prompt examples/foundry_prompt.cpp)
    target_link_libraries(cail_foundry_prompt PRIVATE cail::cail)
endif()

if(BUILD_TESTING)
    add_executable(cail_tests tests/cail_test.cpp)
    target_link_libraries(cail_tests PRIVATE cail::cail)
    add_test(NAME cail_tests COMMAND cail_tests)

    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_executable(cail_local_http_test tests/local_http_test.cpp)
        target_link_libraries(cail_local_http_test PRIVATE cail::cail)
        add_test(NAME cail_local_http
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/local_http_server.py"
                "$<TARGET_FILE:cail_local_http_test>")
    endif()

    add_test(
        NAME cail_install_smoke
        COMMAND "${CMAKE_COMMAND}"
            "-DCAIL_BUILD_DIR=${CMAKE_BINARY_DIR}"
            "-DCAIL_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
            "-DCAIL_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/install-smoke"
            "-DCAIL_TEST_BUILD=${CMAKE_BINARY_DIR}/install-smoke-consumer"
            "-DCAIL_GENERATOR=${CMAKE_GENERATOR}"
            "-DCAIL_CONFIGURATION=$<CONFIG>"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/install-smoke.cmake"
    )
endif()

if(MSVC)
    target_compile_options(cail INTERFACE /Zc:preprocessor)
endif()
