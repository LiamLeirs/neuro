add_executable(neuro_linear_regression linear_regression.cpp)
add_executable(neuro_mnist mnist.cpp)
add_executable(neuro_xor xor.cpp)
add_executable(neuro_cnn mnist_cnn.cpp)

target_link_libraries(neuro_linear_regression PRIVATE Neuro::Neuro)
target_link_libraries(neuro_mnist PRIVATE Neuro::Neuro)
target_link_libraries(neuro_xor PRIVATE Neuro::Neuro)
target_link_libraries(neuro_cnn PRIVATE Neuro::Neuro)