
#include "neuro/Tensor.hpp"

#include <arrayfire.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using neuro::Tensor;

void checkArray(const af::array& actual, const std::vector<float>& expected,
                const std::string& label) {
    if (actual.elements() != expected.size())
        throw std::runtime_error(label + ": incorrect element count");

    std::vector<float> values(expected.size());
    actual.host(values.data());

    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::abs(values[i] - expected[i]) > 1e-5f) {
            throw std::runtime_error(label + ": mismatch at index " + std::to_string(i) +
                                     ", expected " + std::to_string(expected[i]) + ", got " +
                                     std::to_string(values[i]));
        }
    }
}

void testAddition() {
    {
        Tensor x({1, 2, 3}, true);
        Tensor b({2}, true);

        Tensor y = x + b;
        checkArray(y.data(), {3, 4, 5}, "Addition forward");

        y.sum().backward();

        checkArray(x.grad(), {1, 1, 1}, "Addition x.grad");
        checkArray(b.grad(), {3}, "Addition b.grad");
    }

    {
        Tensor b({2}, true);
        Tensor x({1, 2, 3}, true);

        Tensor y = b + x;
        checkArray(y.data(), {3, 4, 5}, "Reverse addition forward");

        y.sum().backward();

        checkArray(b.grad(), {3}, "Reverse addition b.grad");
        checkArray(x.grad(), {1, 1, 1}, "Reverse addition x.grad");
    }

    {
        Tensor x({1, 2, 3}, true);
        Tensor y({4, 5, 6}, true);

        Tensor z = x + y;
        checkArray(z.data(), {5, 7, 9}, "Elementwise addition");

        z.sum().backward();

        checkArray(x.grad(), {1, 1, 1}, "Elementwise addition x.grad");
        checkArray(y.grad(), {1, 1, 1}, "Elementwise addition y.grad");
    }

    std::cout << "[PASS] Addition\n";
}

void testSubtraction() {
    {
        Tensor x({1, 2, 3}, true);
        Tensor b({2}, true);

        Tensor y = x - b;
        checkArray(y.data(), {-1, 0, 1}, "Subtraction forward");

        y.sum().backward();

        checkArray(x.grad(), {1, 1, 1}, "Subtraction x.grad");
        checkArray(b.grad(), {-3}, "Subtraction b.grad");
    }

    {
        Tensor b({2}, true);
        Tensor x({1, 2, 3}, true);

        Tensor y = b - x;
        checkArray(y.data(), {1, 0, -1}, "Reverse subtraction forward");

        y.sum().backward();

        checkArray(b.grad(), {3}, "Reverse subtraction b.grad");
        checkArray(x.grad(), {-1, -1, -1}, "Reverse subtraction x.grad");
    }

    std::cout << "[PASS] Subtraction\n";
}

void testMultiplication() {
    {
        Tensor x({1, 2, 3}, true);
        Tensor w({2}, true);

        Tensor y = x * w;
        checkArray(y.data(), {2, 4, 6}, "Multiplication forward");

        y.sum().backward();

        checkArray(x.grad(), {2, 2, 2}, "Multiplication x.grad");
        checkArray(w.grad(), {6}, "Multiplication w.grad");
    }

    {
        Tensor w({2}, true);
        Tensor x({1, 2, 3}, true);

        Tensor y = w * x;
        checkArray(y.data(), {2, 4, 6}, "Reverse multiplication forward");

        y.sum().backward();

        checkArray(w.grad(), {6}, "Reverse multiplication w.grad");
        checkArray(x.grad(), {2, 2, 2}, "Reverse multiplication x.grad");
    }

    {
        Tensor x({1, 2, 3}, true);
        Tensor y({4, 5, 6}, true);

        Tensor z = x * y;
        checkArray(z.data(), {4, 10, 18}, "Elementwise multiplication");

        z.sum().backward();

        checkArray(x.grad(), {4, 5, 6}, "Elementwise multiplication x.grad");
        checkArray(y.grad(), {1, 2, 3}, "Elementwise multiplication y.grad");
    }

    std::cout << "[PASS] Multiplication\n";
}

void testDivision() {
    {
        Tensor x({2, 4, 6}, true);
        Tensor b({2}, true);

        Tensor y = x / b;
        checkArray(y.data(), {1, 2, 3}, "Division forward");

        y.sum().backward();

        checkArray(x.grad(), {0.5f, 0.5f, 0.5f}, "Division x.grad");
        checkArray(b.grad(), {-3}, "Division b.grad");
    }

    {
        Tensor b({2}, true);
        Tensor x({1, 2, 4}, true);

        Tensor y = b / x;
        checkArray(y.data(), {2, 1, 0.5f}, "Reverse division forward");

        y.sum().backward();

        checkArray(b.grad(), {1.75f}, "Reverse division b.grad");
        checkArray(x.grad(), {-2, -0.5f, -0.125f}, "Reverse division x.grad");
    }

    {
        Tensor x({2, 6, 12}, true);
        Tensor y({2, 3, 4}, true);

        Tensor z = x / y;
        checkArray(z.data(), {1, 2, 3}, "Elementwise division");

        z.sum().backward();

        checkArray(x.grad(), {0.5f, 1.0f / 3, 0.25f}, "Elementwise division x.grad");
        checkArray(y.grad(), {-0.5f, -2.0f / 3, -0.75f}, "Elementwise division y.grad");
    }

    std::cout << "[PASS] Division\n";
}

void testRequiresGrad() {
    Tensor x({1, 2, 3}, false);
    Tensor b({2}, true);

    Tensor y = x * b;
    y.sum().backward();

    checkArray(b.grad(), {6}, "Requires-grad b.grad");
    checkArray(x.grad(), {0, 0, 0}, "Requires-grad x.grad");

    std::cout << "[PASS] Requires-grad handling\n";
}

int main() {
    try {
        testAddition();
        testSubtraction();
        testMultiplication();
        testDivision();
        testRequiresGrad();

        std::cout << "\nAll scalar broadcasting tests passed!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] " << e.what() << '\n';
        return 1;
    }
}
