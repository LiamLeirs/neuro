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
    test_mean.cpp
    test_trainer.cpp
    test_reshape.cpp
)

target_link_libraries(neuro_tests
    PRIVATE
        Neuro::Neuro
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(neuro_tests
    DISCOVERY_MODE PRE_TEST
)
