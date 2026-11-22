
cmake_minimum_required(VERSION 3.20)

project(neuro LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

find_package(ArrayFire REQUIRED)

add_library(neuro
    src/Tensor.cpp
    src/optim/SGD.cpp
    src/data/DataLoader.cpp
    src/nn/Linear.cpp
)

target_include_directories(neuro
    PUBLIC include
)

target_link_libraries(neuro
    PUBLIC ArrayFire::afcuda
)

add_executable(neuro_example
    examples/xor.cpp
)

target_link_libraries(neuro_example
    PRIVATE neuro
)

include(CTest)

if(BUILD_TESTING)
    include(FetchContent)

    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    add_subdirectory(tests)
endif()
