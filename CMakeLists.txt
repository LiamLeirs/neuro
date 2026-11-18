
cmake_minimum_required(VERSION 3.20)

project(neuro LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

find_package(ArrayFire REQUIRED)

add_library(neuro
    src/Tensor.cpp
)

target_include_directories(neuro
    PUBLIC include
)

target_link_libraries(neuro
    PUBLIC ArrayFire::afcuda
)

add_executable(neuro_example
    examples/main.cpp
    include/neuro/TensorImpl.hpp
)

target_link_libraries(neuro_example
    PRIVATE neuro
)
