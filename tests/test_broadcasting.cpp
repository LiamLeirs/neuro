
#include "TestUtils.hpp"
#include "neuro/Tensor.hpp"

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <initializer_list>
#include <stdexcept>
#include <vector>

using namespace neuro;
using neuro::testing::expectArrayNear;

using Matrix = std::initializer_list<std::initializer_list<float>>;

// ==================== Scalar Broadcasting ====================

TEST(BroadcastingScalar, Addition) {
    Tensor x({1, 2, 3}, true);
    Tensor b({10}, true);

    Tensor y = x + b;
    y.sum().backward();

    expectArrayNear(y.data(), af::dim4(3), {11, 12, 13});
    expectArrayNear(x.grad(), af::dim4(3), {1, 1, 1});
    expectArrayNear(b.grad(), af::dim4(1), {3});
}

TEST(BroadcastingScalar, ReverseAddition) {
    Tensor x({1, 2, 3}, true);
    Tensor b({10}, true);

    (b + x).sum().backward();

    expectArrayNear(x.grad(), af::dim4(3), {1, 1, 1});
    expectArrayNear(b.grad(), af::dim4(1), {3});
}

TEST(BroadcastingScalar, Subtraction) {
    Tensor x({1, 2, 3}, true);
    Tensor b({10}, true);

    (x - b).sum().backward();

    expectArrayNear(x.grad(), af::dim4(3), {1, 1, 1});
    expectArrayNear(b.grad(), af::dim4(1), {-3});
}

TEST(BroadcastingScalar, ReverseSubtraction) {
    Tensor x({1, 2, 3}, true);
    Tensor b({10}, true);

    (b - x).sum().backward();

    expectArrayNear(x.grad(), af::dim4(3), {-1, -1, -1});
    expectArrayNear(b.grad(), af::dim4(1), {3});
}

TEST(BroadcastingScalar, Multiplication) {
    Tensor x({1, 2, 3}, true);
    Tensor b({2}, true);

    (x * b).sum().backward();

    expectArrayNear(x.grad(), af::dim4(3), {2, 2, 2});
    expectArrayNear(b.grad(), af::dim4(1), {6});
}

TEST(BroadcastingScalar, ReverseMultiplication) {
    Tensor x({1, 2, 3}, true);
    Tensor b({2}, true);

    (b * x).sum().backward();

    expectArrayNear(x.grad(), af::dim4(3), {2, 2, 2});
    expectArrayNear(b.grad(), af::dim4(1), {6});
}

TEST(BroadcastingScalar, Division) {
    Tensor x({2, 4, 6}, true);
    Tensor b({2}, true);

    (x / b).sum().backward();

    expectArrayNear(x.grad(), af::dim4(3), {0.5f, 0.5f, 0.5f});

    expectArrayNear(b.grad(), af::dim4(1), {-3});
}

TEST(BroadcastingScalar, ReverseDivision) {
    Tensor x({2, 4, 8}, true);
    Tensor b({16}, true);

    (b / x).sum().backward();

    expectArrayNear(x.grad(), af::dim4(3), {-4, -1, -0.25f});

    expectArrayNear(b.grad(), af::dim4(1), {0.875f});
}

// ==================== Matrix Broadcasting ====================

TEST(BroadcastingMatrix, AcrossColumns) {
    Tensor a(Matrix{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}, true);

    Tensor b(Matrix{{10}, {20}, {30}}, true);

    Tensor y = a + b;

    expectArrayNear(y.data(), af::dim4(3, 4), {11, 25, 39, 12, 26, 40, 13, 27, 41, 14, 28, 42});

    y.sum().backward();

    expectArrayNear(a.grad(), af::dim4(3, 4), std::vector<float>(12, 1.0f));

    expectArrayNear(b.grad(), af::dim4(3, 1), {4, 4, 4});
}

TEST(BroadcastingMatrix, AcrossRows) {
    Tensor a(Matrix{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}, true);

    Tensor b(Matrix{{10, 20, 30, 40}}, true);

    Tensor y = a + b;
    y.sum().backward();

    expectArrayNear(y.data(), af::dim4(3, 4), {11, 15, 19, 22, 26, 30, 33, 37, 41, 44, 48, 52});

    expectArrayNear(b.grad(), af::dim4(1, 4), {3, 3, 3, 3});
}

TEST(BroadcastingMatrix, Subtraction) {
    Tensor a(Matrix{{5, 6, 7}, {8, 9, 10}}, true);

    Tensor b(Matrix{{1}, {2}}, true);

    Tensor y = b - a;
    y.sum().backward();

    expectArrayNear(a.grad(), af::dim4(2, 3), std::vector<float>(6, -1.0f));

    expectArrayNear(b.grad(), af::dim4(2, 1), {3, 3});
}

TEST(BroadcastingMatrix, Multiplication) {
    Tensor a(Matrix{{1, 2}, {3, 4}, {5, 6}}, true);

    Tensor b(Matrix{{2}, {3}, {4}}, true);

    Tensor y = a * b;
    y.sum().backward();

    expectArrayNear(a.grad(), af::dim4(3, 2), {2, 3, 4, 2, 3, 4});

    expectArrayNear(b.grad(), af::dim4(3, 1), {3, 7, 11});
}

TEST(BroadcastingMatrix, Division) {
    Tensor a(Matrix{{2, 4, 6}, {8, 12, 16}}, true);

    Tensor b(Matrix{{2}, {4}}, true);

    Tensor y = a / b;
    y.sum().backward();

    expectArrayNear(a.grad(), af::dim4(2, 3), {0.5f, 0.25f, 0.5f, 0.25f, 0.5f, 0.25f});

    expectArrayNear(b.grad(), af::dim4(2, 1), {-3, -2.25f});
}

// ==================== Multiple Broadcast Axes ====================

TEST(BroadcastingMultiAxis, Multiplication) {
    Tensor a(af::constant(1.0f, 2, 3, 4), true);
    Tensor b(af::constant(2.0f, 1, 3, 1), true);

    Tensor y = a * b;
    y.sum().backward();

    expectArrayNear(y.data(), af::dim4(2, 3, 4), std::vector<float>(24, 2.0f));

    expectArrayNear(a.grad(), af::dim4(2, 3, 4), std::vector<float>(24, 2.0f));

    expectArrayNear(b.grad(), af::dim4(1, 3, 1), {8, 8, 8});
}

TEST(BroadcastingMultiAxis, Addition) {
    Tensor a(af::constant(1.0f, 2, 3, 4), true);
    Tensor b(af::constant(5.0f, 1, 3, 1), true);

    (a + b).sum().backward();

    expectArrayNear(a.grad(), af::dim4(2, 3, 4), std::vector<float>(24, 1.0f));

    expectArrayNear(b.grad(), af::dim4(1, 3, 1), {8, 8, 8});
}

// ==================== Validation ====================

TEST(BroadcastingValidation, IncompatibleShapes) {
    Tensor a(af::constant(1.0f, 3, 4), true);
    Tensor b(af::constant(2.0f, 2, 4), true);

    EXPECT_THROW(
        {
            Tensor y = a + b;
            y.data().eval();
        },
        af::exception);
}

TEST(BroadcastingValidation, NoAutograd) {
    Tensor a({1, 2, 3}, false);
    Tensor b({10}, false);

    Tensor y = a + b;

    EXPECT_FALSE(y.requires_grad());

    expectArrayNear(y.data(), af::dim4(3), {11, 12, 13});
}

TEST(BroadcastingValidation, OneOperandRequiresGrad) {
    Tensor a({1, 2, 3}, true);
    Tensor b({10}, false);

    Tensor y = a + b;
    y.sum().backward();

    expectArrayNear(a.grad(), af::dim4(3), {1, 1, 1});

    expectArrayNear(b.grad(), af::dim4(1), {0});
}
