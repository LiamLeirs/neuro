
cmake_minimum_required(VERSION 3.20)

project(neuro VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(ArrayFire REQUIRED)

add_executable(neuro_example
    examples/main.cpp
)

target_link_libraries(neuro_example
    PRIVATE ArrayFire::afcuda
)
