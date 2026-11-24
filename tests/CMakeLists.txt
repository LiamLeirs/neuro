add_executable(neuro_tests
    test_slicing.cpp
    test_tensor.cpp
    test_autograd.cpp
    test_broadcasting.cpp
    test_optim.cpp
    test_dataloader.cpp
    test_index_select.cpp
    test_linear.cpp
    test_activation.cpp
    test_sequential.cpp
    test_adam.cpp
    test_losses.cpp
    test_datasets.cpp
    test_stacking.cpp
    test_mnist.cpp
    test_grad_mode.cpp
    test_tensor_argmax.cpp
    test_accuracy.cpp
)

target_link_libraries(neuro_tests
    PRIVATE
        neuro
        GTest::gtest_main
)

target_compile_features(neuro_tests PRIVATE cxx_std_20)

include(GoogleTest)
gtest_discover_tests(neuro_tests)
