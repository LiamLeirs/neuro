#include "arrayfire.h"
#include "neuro/Tensor.hpp"

#include <af/algorithm.h>
#include <af/array.h>
#include <cassert>
#include <iostream>

using neuro::Tensor;

void checkGrad(const Tensor& tensor, double expected) {
    assert(af::allTrue<bool>(af::abs(tensor.grad() - expected) < 1e-5));
}

int main() {
    Tensor a({4.0}, true);
    Tensor b({2.0}, true);

    // L = ((a * b) + a - b) / b
    Tensor loss = ((a * b) + a - b) / b;

    loss.backward();

    // dL/da = 1 + 1/b = 1.5
    // dL/db = -a/(b*b) = -1.0

    checkGrad(a, 1.5);
    checkGrad(b, -1.0);

    std::cout << "All autograd tests passed!\n";
}