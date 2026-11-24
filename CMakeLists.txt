
cmake_minimum_required(VERSION 3.24)

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
    EXPORT_NAME Neuro
)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

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

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/NeuroConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/NeuroConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Neuro
)

target_include_directories(neuro
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(neuro
    PUBLIC ArrayFire::afcuda
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

option(NEURO_BUILD_EXAMPLES "Build neuro examples" OFF)
option(NEURO_BUILD_TESTS "Build Neuro unit tests" OFF)

if(NEURO_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

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
