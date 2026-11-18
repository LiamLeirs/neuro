
#include "arrayfire.h"
#include "neuro/Tensor.hpp"

#include <af/algorithm.h>
#include <af/array.h>
#include <cassert>
#include <iostream>

using neuro::Tensor;

void check(const Tensor& actual, const af::array& expected) {
    assert(af::allTrue<bool>(af::abs(actual.data() - expected) < 1e-5));
}

int main() {
    Tensor a({2.0f, 4.0f}, true);
    Tensor b({1.0f, 2.0f}, true);

    // Tensor-Tensor arithmetic
    check(a + b, af::array({3.0f, 6.0f}));
    check(a - b, af::array({1.0f, 2.0f}));
    check(a * b, af::array({2.0f, 8.0f}));
    check(a / b, af::array({2.0f, 2.0f}));

    // Tensor-Scalar arithmetic
    check(a + 2.0, af::array({4.0f, 6.0f}));
    check(a - 2.0, af::array({0.0f, 2.0f}));
    check(a * 2.0, af::array({4.0f, 8.0f}));
    check(a / 2.0, af::array({1.0f, 2.0f}));

    // Scalar-Tensor arithmetic
    check(2.0 + a, af::array({4.0f, 6.0f}));
    check(10.0 - a, af::array({8.0f, 6.0f}));
    check(2.0 * a, af::array({4.0f, 8.0f}));
    check(12.0 / a, af::array({6.0f, 3.0f}));

    // Chained operations
    check((a + b) * 2.0, af::array({6.0f, 12.0f}));
    check(10.0 - (a * b), af::array({8.0f, 2.0f}));
    check(12.0 / (a + b), af::array({4.0f, 2.0f}));

    // Gradient tracking
    assert((a + b).requires_grad());
    assert((a - b).requires_grad());
    assert((a * b).requires_grad());
    assert((a / b).requires_grad());
    assert((10.0 - a).requires_grad());
    assert((12.0 / a).requires_grad());

    // No gradient tracking
    Tensor c({2.0f, 4.0f});
    Tensor d({1.0f, 2.0f});

    assert(!(c + d).requires_grad());
    assert(!(c * d).requires_grad());
    assert(!(12.0 / c).requires_grad());

    // Mixed gradient tracking
    assert((a + c).requires_grad());
    assert((c / a).requires_grad());

    std::cout << "All arithmetic tests passed!\n";
}
