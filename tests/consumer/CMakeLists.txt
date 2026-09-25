cmake_minimum_required(VERSION 3.31)

project(cail_package_consumer LANGUAGES CXX)

find_package(cail 0.1 CONFIG REQUIRED)

add_executable(cail_package_consumer main.cpp)
target_link_libraries(cail_package_consumer PRIVATE cail::cail)

enable_testing()
add_test(NAME cail_package_consumer COMMAND cail_package_consumer)
