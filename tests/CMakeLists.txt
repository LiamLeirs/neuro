add_executable(neuro_tests
    test_slicing.cpp
    test_tensor.cpp
    test_autograd.cpp
)

target_link_libraries(neuro_tests
    PRIVATE
        neuro
        GTest::gtest_main
)

target_compile_features(neuro_tests PRIVATE cxx_std_20)

include(GoogleTest)
gtest_discover_tests(neuro_tests)
