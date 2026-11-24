add_executable(neuro_linear_regression linear_regression.cpp)
add_executable(neuro_mnist mnist.cpp)
add_executable(neuro_xor xor.cpp)

target_link_libraries(neuro_linear_regression PRIVATE Neuro::Neuro)
target_link_libraries(neuro_mnist PRIVATE Neuro::Neuro)
target_link_libraries(neuro_xor PRIVATE Neuro::Neuro)