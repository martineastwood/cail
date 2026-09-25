cmake_minimum_required(VERSION 3.31)

project(cail VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

# HTTPS is required by provider clients.
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
FetchContent_MakeAvailable(magic_enum)

add_library(cail INTERFACE)
add_library(cail::cail ALIAS cail)

target_compile_features(cail INTERFACE cxx_std_23)
target_include_directories(cail INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cail INTERFACE
    glaze::glaze
    glaze::asio
    magic_enum::magic_enum
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
endif()

if(MSVC)
    target_compile_options(cail INTERFACE /Zc:preprocessor)
endif()
