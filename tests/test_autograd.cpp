
#include "TestUtils.hpp"
#include "neuro/Tensor.hpp"

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <initializer_list>
#include <stdexcept>

using namespace neuro;
using neuro::testing::expectArrayNear;

using Matrix = std::initializer_list<std::initializer_list<float>>;

// ==================== Basic Gradients ====================

TEST(Autograd, Addition) {
    Tensor a({1, 2, 3}, true);
    Tensor b({4, 5, 6}, true);

    (a + b).sum().backward();

    expectArrayNear(a.grad(), af::dim4(3), {1, 1, 1});
    expectArrayNear(b.grad(), af::dim4(3), {1, 1, 1});
}

TEST(Autograd, Subtraction) {
    Tensor a({1, 2, 3}, true);
    Tensor b({4, 5, 6}, true);

    (a - b).sum().backward();

    expectArrayNear(a.grad(), af::dim4(3), {1, 1, 1});
    expectArrayNear(b.grad(), af::dim4(3), {-1, -1, -1});
}

TEST(Autograd, Multiplication) {
    Tensor a({1, 2, 3}, true);
    Tensor b({4, 5, 6}, true);

    (a * b).sum().backward();

    expectArrayNear(a.grad(), af::dim4(3), {4, 5, 6});
    expectArrayNear(b.grad(), af::dim4(3), {1, 2, 3});
}

TEST(Autograd, Division) {
    Tensor a({2, 4, 6}, true);
    Tensor b({2, 2, 3}, true);

    (a / b).sum().backward();

    expectArrayNear(a.grad(), af::dim4(3), {0.5f, 0.5f, 1.0f / 3.0f});

    expectArrayNear(b.grad(), af::dim4(3), {-0.5f, -1.0f, -2.0f / 3.0f});
}

// ==================== Scalar Gradients ====================

TEST(AutogradScalar, Multiplication) {
    Tensor x({1, 2, 3}, true);

    (x * 3.0).sum().backward();

    expectArrayNear(x.grad(), af::dim4(3), {3, 3, 3});
}

TEST(AutogradScalar, Division) {
    Tensor x({2, 4, 6}, true);

    (x / 2.0).sum().backward();

    expectArrayNear(x.grad(), af::dim4(3), {0.5f, 0.5f, 0.5f});
}

TEST(AutogradScalar, ReverseSubtraction) {
    Tensor x({1, 2, 3}, true);

    (10.0 - x).sum().backward();

    expectArrayNear(x.grad(), af::dim4(3), {-1, -1, -1});
}

TEST(AutogradScalar, ReverseDivision) {
    Tensor x({2, 4, 8}, true);

    (16.0 / x).sum().backward();

    expectArrayNear(x.grad(), af::dim4(3), {-4.0f, -1.0f, -0.25f});
}

// ==================== Computational Graphs ====================

TEST(AutogradGraph, ChainRule) {
    Tensor x({2.0f}, true);

    Tensor y = x * x;
    Tensor z = y * 3.0;

    z.backward();

    // z = 3x², dz/dx = 6x = 12
    expectArrayNear(x.grad(), af::dim4(1), {12});
}

TEST(AutogradGraph, SharedDependency) {
    Tensor x({2.0f}, true);

    Tensor y = x * x;
    Tensor z = y + y;

    z.backward();

    // z = 2x², dz/dx = 4x = 8
    expectArrayNear(x.grad(), af::dim4(1), {8});
}

TEST(AutogradGraph, Branching) {
    Tensor x({3.0f}, true);

    Tensor a = x * 2.0;
    Tensor b = x * x;
    Tensor loss = a + b;

    loss.backward();

    // d(2x + x²)/dx = 2 + 2x = 8
    expectArrayNear(x.grad(), af::dim4(1), {8});
}

TEST(AutogradGraph, NoGradPropagation) {
    Tensor a({2.0f}, true);
    Tensor b({3.0f}, false);

    (a * b).backward();

    expectArrayNear(a.grad(), af::dim4(1), {3});
    expectArrayNear(b.grad(), af::dim4(1), {0});
}

TEST(AutogradGraph, ZeroGrad) {
    Tensor x({2.0f}, true);

    (x * x).backward();
    expectArrayNear(x.grad(), af::dim4(1), {4});

    x.zeroGrad();
    expectArrayNear(x.grad(), af::dim4(1), {0});
}

TEST(AutogradGraph, NonScalarBackwardRejected) {
    Tensor x({1, 2, 3}, true);

    EXPECT_THROW(x.backward(), std::invalid_argument);
}

// ==================== Matrix Gradients ====================

TEST(AutogradMatrix, Matmul) {
    Tensor a(Matrix{{1, 2}, {3, 4}}, true);

    Tensor b(Matrix{{5, 6}, {7, 8}}, true);

    a.matmul(b).sum().backward();

    expectArrayNear(a.grad(), af::dim4(2, 2), {11, 11, 15, 15});

    expectArrayNear(b.grad(), af::dim4(2, 2), {4, 6, 4, 6});
}

TEST(AutogradMatrix, Transpose) {
    Tensor x(Matrix{{1, 2, 3}, {4, 5, 6}}, true);

    x.transpose().sum().backward();

    expectArrayNear(x.grad(), af::dim4(2, 3), {1, 1, 1, 1, 1, 1});
}

TEST(AutogradMatrix, SumDimensionZero) {
    Tensor x(Matrix{{1, 2, 3}, {4, 5, 6}}, true);

    x.sum(0).sum().backward();

    expectArrayNear(x.grad(), af::dim4(2, 3), {1, 1, 1, 1, 1, 1});
}

TEST(AutogradMatrix, SumDimensionOne) {
    Tensor x(Matrix{{1, 2, 3}, {4, 5, 6}}, true);

    x.sum(1).sum().backward();

    expectArrayNear(x.grad(), af::dim4(2, 3), {1, 1, 1, 1, 1, 1});
}

TEST(AutogradMatrix, Indexing) {
    Tensor x({2, 4, 6}, true);

    Tensor loss = x[1] * x[1];
    loss.backward();

    expectArrayNear(x.grad(), af::dim4(3), {0, 8, 0});
}

TEST(AutogradMean, BackwardVector) {
    Tensor x({1, 2, 3, 4}, true);

    x.mean().backward();

    expectArrayNear(x.grad(), af::dim4(4), {0.25f, 0.25f, 0.25f, 0.25f});
}

TEST(AutogradMean, BackwardMatrix) {
    Tensor x(Matrix{{1, 2, 3}, {4, 5, 6}}, true);

    x.mean().backward();

    expectArrayNear(x.grad(), af::dim4(2, 3),
                    {1.0f / 6, 1.0f / 6, 1.0f / 6, 1.0f / 6, 1.0f / 6, 1.0f / 6});
}

TEST(AutogradMean, BackwardDimensionZero) {
    Tensor x(Matrix{{1, 2, 3}, {4, 5, 6}}, true);

    x.mean(0).sum().backward();

    // Two rows: each element receives 1/2.
    expectArrayNear(x.grad(), af::dim4(2, 3), {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f});
}

TEST(AutogradMean, BackwardDimensionOne) {
    Tensor x(Matrix{{1, 2, 3}, {4, 5, 6}}, true);

    x.mean(1).sum().backward();

    // Three columns: each element receives 1/3.
    expectArrayNear(x.grad(), af::dim4(2, 3),
                    {1.0f / 3, 1.0f / 3, 1.0f / 3, 1.0f / 3, 1.0f / 3, 1.0f / 3});
}

TEST(AutogradMean, ChainRule) {
    Tensor x({1, 2, 3, 4}, true);

    Tensor y = (x * 2.0).mean();
    y.backward();

    expectArrayNear(x.grad(), af::dim4(4), {0.5f, 0.5f, 0.5f, 0.5f});
}

TEST(AutogradMean, DimensionMeanChainRule) {
    Tensor x(Matrix{{1, 2, 3}, {4, 5, 6}}, true);

    Tensor y = (x * 3.0).mean(0).sum();
    y.backward();

    // Derivative: 3 / 2 = 1.5.
    expectArrayNear(x.grad(), af::dim4(2, 3), {1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f});
}

TEST(AutogradMean, GradientAccumulation) {
    Tensor x({1, 2, 3, 4}, true);

    Tensor y = x.mean() + x.sum();
    y.backward();

    expectArrayNear(x.grad(), af::dim4(4), {1.25f, 1.25f, 1.25f, 1.25f});
}

TEST(AutogradExp, Forward) {
    Tensor x({0.0f, 1.0f, 2.0f});

    Tensor result = x.exp();

    expectArrayNear(result.data(), af::dim4(3), {1.0f, std::exp(1.0f), std::exp(2.0f)});
}

TEST(AutogradExp, Backward) {
    Tensor x({0.0f, 1.0f, 2.0f}, true);

    x.exp().sum().backward();

    expectArrayNear(x.grad(), af::dim4(3), {1.0f, std::exp(1.0f), std::exp(2.0f)});
}

TEST(AutogradLog, Forward) {
    Tensor x({1.0f, 2.0f, 4.0f});

    Tensor result = x.log();

    expectArrayNear(result.data(), af::dim4(3), {0.0f, std::log(2.0f), std::log(4.0f)});
}

TEST(AutogradLog, Backward) {
    Tensor x({1.0f, 2.0f, 4.0f}, true);

    x.log().sum().backward();

    expectArrayNear(x.grad(), af::dim4(3), {1.0f, 0.5f, 0.25f});
}

TEST(AutogradMax, GlobalForwardBackward) {
    Tensor x({1.0f, 5.0f, 3.0f, 2.0f}, true);

    Tensor result = x.max();

    EXPECT_NEAR(result.data().scalar<float>(), 5.0f, 1e-5f);

    result.backward();

    expectArrayNear(x.grad(), af::dim4(4), {0.0f, 1.0f, 0.0f, 0.0f});
}

TEST(AutogradMax, GlobalTiedMaxima) {
    Tensor x({2.0f, 5.0f, 5.0f, 1.0f}, true);

    x.max().backward();

    expectArrayNear(x.grad(), af::dim4(4), {0.0f, 0.5f, 0.5f, 0.0f});
}

TEST(AutogradMax, DimensionForwardBackward) {
    Tensor x(Matrix{{1.0f, 5.0f, 3.0f}, {4.0f, 2.0f, 6.0f}}, true);

    Tensor result = x.max(1);

    expectArrayNear(result.data(), af::dim4(2, 1), {5.0f, 6.0f});

    result.sum().backward();

    expectArrayNear(x.grad(), af::dim4(2, 3), {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f});
}

TEST(AutogradMax, DimensionTiedMaxima) {
    Tensor x(Matrix{{2.0f, 5.0f, 5.0f}, {4.0f, 4.0f, 1.0f}}, true);

    x.max(1).sum().backward();

    expectArrayNear(x.grad(), af::dim4(2, 3), {0.0f, 0.5f, 0.5f, 0.5f, 0.5f, 0.0f});
}

TEST(AutogradMax, NonUniformUpstreamGradients) {
    Tensor x(Matrix{{1.0f, 5.0f, 3.0f}, {4.0f, 2.0f, 6.0f}}, true);

    Tensor result = x.max(1);

    // Different gradient magnitude for each row.
    Tensor weights({2.0f, 3.0f});

    (result * weights).sum().backward();

    expectArrayNear(x.grad(), af::dim4(2, 3), {0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 3.0f});
}
