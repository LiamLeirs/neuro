
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

// ==================== Constructors ====================

TEST(Tensor, VectorConstruction) {
    Tensor x({1.0f, 2.0f, 3.0f}, true);

    expectArrayNear(x.data(), af::dim4(3), {1, 2, 3});

    EXPECT_TRUE(x.requires_grad());
}

TEST(Tensor, MatrixConstruction) {
    Tensor x(Matrix{{1, 2, 3}, {4, 5, 6}}, true);

    expectArrayNear(x.data(), af::dim4(2, 3), {1, 4, 2, 5, 3, 6});
}

TEST(Tensor, ArrayFireConstruction) {
    af::array data = af::constant(5.0f, 2, 3);

    Tensor x(data, false);

    expectArrayNear(x.data(), af::dim4(2, 3), {5, 5, 5, 5, 5, 5});

    EXPECT_FALSE(x.requires_grad());
}

TEST(Tensor, InitialGradientIsZero) {
    Tensor x({1.0f, 2.0f, 3.0f}, true);

    expectArrayNear(x.grad(), af::dim4(3), {0, 0, 0});
}

TEST(Tensor, InvalidMatrixConstruction) {
    EXPECT_THROW((Tensor(Matrix{{1, 2}, {3}})), std::invalid_argument);
}

// ==================== Tensor Arithmetic ====================

TEST(TensorArithmetic, Addition) {
    Tensor a({1, 2, 3});
    Tensor b({4, 5, 6});

    Tensor result = a + b;

    expectArrayNear(result.data(), af::dim4(3), {5, 7, 9});
}

TEST(TensorArithmetic, Subtraction) {
    Tensor a({5, 7, 9});
    Tensor b({1, 2, 3});

    Tensor result = a - b;

    expectArrayNear(result.data(), af::dim4(3), {4, 5, 6});
}

TEST(TensorArithmetic, Multiplication) {
    Tensor a({2, 3, 4});
    Tensor b({5, 6, 7});

    Tensor result = a * b;

    expectArrayNear(result.data(), af::dim4(3), {10, 18, 28});
}

TEST(TensorArithmetic, Division) {
    Tensor a({10, 20, 30});
    Tensor b({2, 4, 5});

    Tensor result = a / b;

    expectArrayNear(result.data(), af::dim4(3), {5, 5, 6});
}

// ==================== Scalar Arithmetic ====================

TEST(TensorScalar, Addition) {
    Tensor x({1, 2, 3});

    expectArrayNear((x + 2.0).data(), af::dim4(3), {3, 4, 5});

    expectArrayNear((2.0 + x).data(), af::dim4(3), {3, 4, 5});
}

TEST(TensorScalar, Subtraction) {
    Tensor x({1, 2, 3});

    expectArrayNear((x - 2.0).data(), af::dim4(3), {-1, 0, 1});

    expectArrayNear((10.0 - x).data(), af::dim4(3), {9, 8, 7});
}

TEST(TensorScalar, Multiplication) {
    Tensor x({1, 2, 3});

    expectArrayNear((x * 3.0).data(), af::dim4(3), {3, 6, 9});

    expectArrayNear((3.0 * x).data(), af::dim4(3), {3, 6, 9});
}

TEST(TensorScalar, Division) {
    Tensor x({2, 4, 8});

    expectArrayNear((x / 2.0).data(), af::dim4(3), {1, 2, 4});

    expectArrayNear((16.0 / x).data(), af::dim4(3), {8, 4, 2});
}

// ==================== Matrix Operations ====================

TEST(TensorMatrix, Matmul) {
    Tensor a(Matrix{{1, 2}, {3, 4}});

    Tensor b(Matrix{{5, 6}, {7, 8}});

    Tensor result = a.matmul(b);

    // [[19,22],
    //  [43,50]]
    expectArrayNear(result.data(), af::dim4(2, 2), {19, 43, 22, 50});
}

TEST(TensorMatrix, Transpose) {
    Tensor x(Matrix{{1, 2, 3}, {4, 5, 6}});

    Tensor result = x.transpose();

    expectArrayNear(result.data(), af::dim4(3, 2), {1, 2, 3, 4, 5, 6});
}

// ==================== Reductions ====================

TEST(TensorReduction, SumAll) {
    Tensor x(Matrix{{1, 2, 3}, {4, 5, 6}});

    Tensor result = x.sum();

    EXPECT_NEAR(result.data().scalar<float>(), 21.0f, 1e-5f);
}

TEST(TensorReduction, SumDimensionZero) {
    Tensor x(Matrix{{1, 2, 3}, {4, 5, 6}});

    Tensor result = x.sum(0);

    expectArrayNear(result.data(), af::dim4(1, 3), {5, 7, 9});
}

TEST(TensorReduction, SumDimensionOne) {
    Tensor x(Matrix{{1, 2, 3}, {4, 5, 6}});

    Tensor result = x.sum(1);

    expectArrayNear(result.data(), af::dim4(2, 1), {6, 15});
}

// ==================== Indexing ====================

TEST(TensorIndexing, ValidIndex) {
    Tensor x({10, 20, 30});

    Tensor result = x[1];

    EXPECT_NEAR(result.data().scalar<float>(), 20.0f, 1e-5f);
}

TEST(TensorIndexing, NegativeIndex) {
    Tensor x({10, 20, 30});

    EXPECT_THROW(x[-1], std::out_of_range);
}

TEST(TensorIndexing, OutOfBounds) {
    Tensor x({10, 20, 30});

    EXPECT_THROW(x[3], std::out_of_range);
}

TEST(TensorIndexing, MatrixIndexingRejected) {
    Tensor x(Matrix{{1, 2}, {3, 4}});

    EXPECT_THROW(x[0], std::invalid_argument);
}

// ==================== Gradient Tracking ====================

TEST(TensorGradTracking, BothDisabled) {
    Tensor a({1, 2, 3}, false);
    Tensor b({4, 5, 6}, false);

    EXPECT_FALSE((a + b).requires_grad());
}

TEST(TensorGradTracking, OneEnabled) {
    Tensor a({1, 2, 3}, true);
    Tensor b({4, 5, 6}, false);

    EXPECT_TRUE((a + b).requires_grad());
}

TEST(TensorGradTracking, BothEnabled) {
    Tensor a({1, 2, 3}, true);
    Tensor b({4, 5, 6}, true);

    EXPECT_TRUE((a * b).requires_grad());
}
