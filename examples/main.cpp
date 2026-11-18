
#include "neuro/Tensor.hpp"

#include <af/algorithm.h>
#include <cassert>
#include <iostream>

int main() {
    neuro::Tensor a({1.0f, 2.0f, 3.0f}, true);
    neuro::Tensor b = a;
    neuro::Tensor c = a * 2.0;

    assert(&a.data() == &b.data()); // Shared state
    assert(&a.grad() == &b.grad()); // Shared gradients
    assert(&a.data() != &c.data()); // Independent result
    assert(c.requires_grad());      // Gradient tracking
    assert(af::allTrue<bool>(a.grad() == 0));

    std::cout << "All tests passed!\n";
}
