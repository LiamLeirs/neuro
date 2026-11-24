
cmake_minimum_required(VERSION 3.20)

project(neuro LANGUAGES CXX VERSION 0.1.0)

find_package(ArrayFire REQUIRED)

add_library(neuro
    src/Tensor.cpp
    src/optim/SGD.cpp
    src/data/DataLoader.cpp
    src/nn/Linear.cpp
    src/nn/Sequential.cpp
    src/optim/Adam.cpp
    src/nn/loss/MSELoss.cpp
    src/nn/loss/MAELoss.cpp
    src/nn/loss/CrossEntropyLoss.cpp
    src/data/TensorDataset.cpp
    src/data/MNISTDataset.cpp
    src/metrics/Accuracy.cpp
    src/metrics/Mean.cpp
    src/Trainer.cpp
)

add_library(Neuro::Neuro ALIAS neuro)

target_compile_features(neuro PUBLIC cxx_std_20)

set_target_properties(neuro PROPERTIES
    CXX_EXTENSIONS OFF
)

target_include_directories(neuro
    PUBLIC include
)

target_link_libraries(neuro
    PUBLIC ArrayFire::afcuda
)

add_executable(neuro_example
    examples/mnist.cpp
)

target_link_libraries(neuro_example
    PRIVATE Neuro::Neuro
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
