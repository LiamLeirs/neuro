cmake_minimum_required(VERSION 3.24)

project(neuro LANGUAGES CXX VERSION 0.2.0)

# Configuration
set(NEURO_BACKEND "CUDA" CACHE STRING "ArrayFire backend (CUDA or CPU)")
set_property(CACHE NEURO_BACKEND PROPERTY STRINGS CUDA CPU)

option(NEURO_BUILD_EXAMPLES "Build Neuro examples" OFF)
option(NEURO_BUILD_TESTS "Build Neuro unit tests" OFF)

# Dependencies
find_package(ArrayFire REQUIRED)

if(NEURO_BACKEND STREQUAL "CUDA")
    set(NEURO_ARRAYFIRE_TARGET ArrayFire::afcuda)
elseif(NEURO_BACKEND STREQUAL "CPU")
    set(NEURO_ARRAYFIRE_TARGET ArrayFire::afcpu)
else()
    message(FATAL_ERROR
        "Unsupported NEURO_BACKEND: ${NEURO_BACKEND}. "
        "Supported backends are CUDA and CPU."
    )
endif()

if(NOT TARGET ${NEURO_ARRAYFIRE_TARGET})
    message(FATAL_ERROR
        "The selected ArrayFire backend (${NEURO_BACKEND}) is not available."
    )
endif()

message(STATUS "Neuro backend: ${NEURO_BACKEND}")

# Library
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
    src/nn/Conv2D.cpp
    src/data/CIFAR10Dataset.cpp
    src/viz/SvgDocument.cpp
    src/viz/Plot.cpp
    src/nn/BatchNorm1D.cpp
    src/nn/BatchNorm2D.cpp
    src/nn/BatchNormBase.cpp
    src/optim/AdamBase.cpp
    src/optim/AdamW.cpp
    src/data/transforms/Compose.cpp
    src/data/TransformDataset.cpp
    src/data/transforms/Normalize.cpp
    src/data/transforms/RandomHorizontalFlip.cpp
    src/data/transforms/RandomCrop.cpp
)

add_library(Neuro::Neuro ALIAS neuro)

target_compile_features(neuro PUBLIC cxx_std_20)

set_target_properties(neuro PROPERTIES
    CXX_EXTENSIONS OFF
    EXPORT_NAME Neuro
)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

target_include_directories(neuro
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(neuro
    PUBLIC ${NEURO_ARRAYFIRE_TARGET}
)

# Package configuration
configure_package_config_file(
    cmake/NeuroConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/NeuroConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Neuro
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/NeuroConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion
)

# Installation
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/NeuroConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/NeuroConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Neuro
)

install(TARGETS neuro
    EXPORT NeuroTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT NeuroTargets
    FILE NeuroTargets.cmake
    NAMESPACE Neuro::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Neuro
)

# Examples
if(NEURO_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Tests
if(NEURO_BUILD_TESTS)
    enable_testing()
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