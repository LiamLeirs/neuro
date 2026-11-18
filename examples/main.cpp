
#include "neuro/Tensor.hpp"

#include <af/algorithm.h>
#include <af/arith.h>
#include <af/array.h>
#include <iostream>

using neuro::Tensor;

void checkGrad(const char* name, const Tensor& actual, const Tensor& expected) {
    std::cout << "\n=== Gradient " << name << " ===\n";

    std::cout << "Actual:\n" << Tensor(actual.grad()) << '\n';
    std::cout << "Expected:\n" << expected << '\n';

    if (actual.grad().dims() != expected.data().dims()) {
        std::cout << "FAIL: Shape mismatch!\n";
        return;
    }

    bool passed = af::allTrue<bool>(af::abs(actual.grad() - expected.data()) < 1e-5);

    std::cout << (passed ? "PASS" : "FAIL") << '\n';
}

int main() {
    // A: 2x3
    Tensor A({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}, true);

    // B: 3x2
    Tensor B({{2.0, 1.0}, {0.0, 3.0}, {4.0, 2.0}}, true);

    // C = A @ B
    Tensor C = A.matmul(B);

    // L = 2 * sum(C^T)
    Tensor loss = C.transpose().sum() * 2.0;

    std::cout << "=== Forward pass ===\n";
    std::cout << "C:\n" << C << '\n';
    std::cout << "Loss:\n" << loss << '\n';

    loss.backward();

    // dL/dA = 2 * ones(2,2) @ B^T
    Tensor expectedA({{6.0, 6.0, 12.0}, {6.0, 6.0, 12.0}});

    // dL/dB = 2 * A^T @ ones(2,2)
    Tensor expectedB({{10.0, 10.0}, {14.0, 14.0}, {18.0, 18.0}});

    checkGrad("A", A, expectedA);
    checkGrad("B", B, expectedB);

    std::cout << "\nExpected loss: 180\n";

    return 0;
}
