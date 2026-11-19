
#include "../include/neuro/Tensor.hpp"

#include <arrayfire.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using neuro::Tensor;

void checkArray(const af::array& actual, const std::vector<float>& expected,
                const std::string& name) {
    if (actual.elements() != expected.size()) {
        throw std::runtime_error(name + ": incorrect element count");
    }

    std::vector<float> values(expected.size());
    actual.host(values.data());

    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::abs(values[i] - expected[i]) > 1e-5f) {
            throw std::runtime_error(name + ": mismatch at index " + std::to_string(i) +
                                     " (expected " + std::to_string(expected[i]) + ", got " +
                                     std::to_string(values[i]) + ")");
        }
    }
}

void testVectorTimesScalar() {
    Tensor x({1.0f, 2.0f, 3.0f}, true);
    Tensor w({2.0f}, true);

    Tensor y = x * w;

    checkArray(y.data(), {2.0f, 4.0f, 6.0f}, "Forward");

    Tensor loss = y.sum();
    loss.backward();

    checkArray(x.grad(), {2.0f, 2.0f, 2.0f}, "x.grad");
    checkArray(w.grad(), {6.0f}, "w.grad");

    std::cout << "[PASS] Vector * scalar\n";
}

void testScalarTimesVector() {
    Tensor w({2.0f}, true);
    Tensor x({1.0f, 2.0f, 3.0f}, true);

    Tensor y = w * x;

    checkArray(y.data(), {2.0f, 4.0f, 6.0f}, "Forward");

    Tensor loss = y.sum();
    loss.backward();

    checkArray(w.grad(), {6.0f}, "w.grad");
    checkArray(x.grad(), {2.0f, 2.0f, 2.0f}, "x.grad");

    std::cout << "[PASS] Scalar * vector\n";
}

void testVectorTimesVector() {
    Tensor x({1.0f, 2.0f, 3.0f}, true);
    Tensor y({4.0f, 5.0f, 6.0f}, true);

    Tensor z = x * y;

    checkArray(z.data(), {4.0f, 10.0f, 18.0f}, "Forward");

    Tensor loss = z.sum();
    loss.backward();

    checkArray(x.grad(), {4.0f, 5.0f, 6.0f}, "x.grad");
    checkArray(y.grad(), {1.0f, 2.0f, 3.0f}, "y.grad");

    std::cout << "[PASS] Vector * vector\n";
}

int main() {
    try {
        testVectorTimesScalar();
        testScalarTimesVector();
        testVectorTimesVector();

        std::cout << "\nAll multiplication broadcasting tests passed!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] " << e.what() << '\n';
        return 1;
    }
}
