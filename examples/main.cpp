
#include "neuro/Tensor.hpp"
#include "neuro/optim/SGD.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace neuro;

bool approximatelyEqual(float a, float b, float epsilon = 1e-5f) {
    return std::abs(a - b) < epsilon;
}

int main() {
    Tensor w({5.0f}, true);
    SGD optimizer({w}, 0.1);

    // Forward pass: L = w^2
    Tensor loss = w * w;

    // Backpropagation: dL/dw = 2w = 10
    loss.backward();

    assert(approximatelyEqual(w.grad().scalar<float>(), 10.0f));

    // SGD: w = 5 - 0.1 * 10 = 4
    optimizer.step();

    assert(approximatelyEqual(w.data().scalar<float>(), 4.0f));

    // SGD should not modify the gradient
    assert(approximatelyEqual(w.grad().scalar<float>(), 10.0f));

    // Reset gradients
    optimizer.zeroGrad();

    assert(approximatelyEqual(w.grad().scalar<float>(), 0.0f));

    std::cout << "All SGD tests passed!\n";

    return 0;
}
