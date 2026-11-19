
#include "../include/neuro/Tensor.hpp"

#include <arrayfire.h>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using neuro::Tensor;

void checkArray(const af::array& actual, const std::vector<float>& expected) {
    assert(actual.elements() == expected.size());

    std::vector<float> values(expected.size());
    actual.host(values.data());

    for (size_t i = 0; i < expected.size(); ++i) {
        assert(std::abs(values[i] - expected[i]) < 1e-5f);
    }
}

void testVectorPlusScalar() {
    Tensor x({1.0f, 2.0f, 3.0f}, true);
    Tensor b({2.0f}, true);

    Tensor y = x + b;
    checkArray(y.data(), {3.0f, 4.0f, 5.0f});

    Tensor loss = y.sum();
    loss.backward();

    checkArray(x.grad(), {1.0f, 1.0f, 1.0f});
    checkArray(b.grad(), {3.0f});

    std::cout << "[PASS] Vector + scalar\n";
}

void testScalarPlusVector() {
    Tensor b({2.0f}, true);
    Tensor x({1.0f, 2.0f, 3.0f}, true);

    Tensor y = b + x;
    checkArray(y.data(), {3.0f, 4.0f, 5.0f});

    Tensor loss = y.sum();
    loss.backward();

    checkArray(b.grad(), {3.0f});
    checkArray(x.grad(), {1.0f, 1.0f, 1.0f});

    std::cout << "[PASS] Scalar + vector\n";
}

void testVectorPlusVector() {
    Tensor x({1.0f, 2.0f, 3.0f}, true);
    Tensor y({4.0f, 5.0f, 6.0f}, true);

    Tensor z = x + y;
    checkArray(z.data(), {5.0f, 7.0f, 9.0f});

    Tensor loss = z.sum();
    loss.backward();

    checkArray(x.grad(), {1.0f, 1.0f, 1.0f});
    checkArray(y.grad(), {1.0f, 1.0f, 1.0f});

    std::cout << "[PASS] Vector + vector\n";
}

int main() {
    try {
        testVectorPlusScalar();
        testScalarPlusVector();
        testVectorPlusVector();

        std::cout << "\nAll broadcasting tests passed!\n";
        return 0;
    } catch (const af::exception& e) {
        std::cerr << "ArrayFire error: " << e.what() << '\n';
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << '\n';
        return 1;
    }
}
