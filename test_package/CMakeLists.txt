cmake_minimum_required(VERSION 3.23)

project(cail_conan_test_package LANGUAGES CXX)

find_package(cail CONFIG REQUIRED)

add_executable(cail_conan_example example.cpp)
target_link_libraries(cail_conan_example PRIVATE cail::cail)
