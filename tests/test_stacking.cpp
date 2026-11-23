#include "TestUtils.hpp"
#include "neuro/Tensor.hpp"

#include <af/array.h>
#include <af/data.h>
#include <af/dim4.hpp>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

using namespace neuro;
using neuro::testing::expectArrayNear;

TEST(TensorStackTest, StackVectorsDim0) {
    Tensor a({1.0f, 2.0f, 3.0f}, true);
    Tensor b({4.0f, 5.0f, 6.0f}, true);

    Tensor result = Tensor::stack({a, b}, 0);

    EXPECT_EQ(result.shape(), af::dim4(2, 3));

    expectArrayNear(result.data(), af::dim4(2, 3), {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f});

    result.sum().backward();

    expectArrayNear(a.grad(), af::dim4(3), {1.0f, 1.0f, 1.0f});
    expectArrayNear(b.grad(), af::dim4(3), {1.0f, 1.0f, 1.0f});
}

TEST(TensorStackTest, StackVectorsDim1) {
    Tensor a({1.0f, 2.0f, 3.0f}, true);
    Tensor b({4.0f, 5.0f, 6.0f}, true);

    Tensor result = Tensor::stack({a, b}, 1);

    EXPECT_EQ(result.shape(), af::dim4(3, 2));

    expectArrayNear(result.data(), af::dim4(3, 2), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    result.sum().backward();

    expectArrayNear(a.grad(), af::dim4(3), {1.0f, 1.0f, 1.0f});
    expectArrayNear(b.grad(), af::dim4(3), {1.0f, 1.0f, 1.0f});
}

TEST(TensorStackTest, StackMatricesDim0) {
    Tensor a({{1.0f, 2.0f}, {3.0f, 4.0f}}, true);

    Tensor b({{5.0f, 6.0f}, {7.0f, 8.0f}}, true);

    Tensor result = Tensor::stack({a, b}, 0);

    EXPECT_EQ(result.shape(), af::dim4(2, 2, 2));

    expectArrayNear(result.data(), af::dim4(2, 2, 2),
                    {1.0f, 5.0f, 3.0f, 7.0f, 2.0f, 6.0f, 4.0f, 8.0f});

    result.sum().backward();

    expectArrayNear(a.grad(), af::dim4(2, 2), {1.0f, 1.0f, 1.0f, 1.0f});

    expectArrayNear(b.grad(), af::dim4(2, 2), {1.0f, 1.0f, 1.0f, 1.0f});
}

TEST(TensorStackTest, RepeatedParentAccumulatesGradients) {
    Tensor a({1.0f, 2.0f, 3.0f}, true);

    Tensor result = Tensor::stack({a, a}, 0);

    result.sum().backward();

    expectArrayNear(a.grad(), af::dim4(3), {2.0f, 2.0f, 2.0f});
}

TEST(TensorStackTest, MixedGradientRequirements) {
    Tensor a({1.0f, 2.0f}, true);
    Tensor b({3.0f, 4.0f}, false);

    Tensor result = Tensor::stack({a, b}, 0);

    EXPECT_TRUE(result.requires_grad());

    result.sum().backward();

    expectArrayNear(a.grad(), af::dim4(2), {1.0f, 1.0f});
    expectArrayNear(b.grad(), af::dim4(2), {0.0f, 0.0f});
}

TEST(TensorStackTest, SingleTensor) {
    Tensor a({1.0f, 2.0f, 3.0f}, true);

    Tensor result = Tensor::stack({a}, 0);

    EXPECT_EQ(result.shape(), af::dim4(1, 3));

    result.sum().backward();

    expectArrayNear(a.grad(), af::dim4(3), {1.0f, 1.0f, 1.0f});
}

TEST(TensorStackTest, RejectsEmptyInput) {
    EXPECT_THROW(Tensor::stack({}, 0), std::invalid_argument);
}

TEST(TensorStackTest, RejectsMismatchedShapes) {
    Tensor a({1.0f, 2.0f});
    Tensor b({3.0f, 4.0f, 5.0f});

    EXPECT_THROW(Tensor::stack({a, b}, 0), std::invalid_argument);
}

TEST(TensorStackTest, RejectsInvalidDimension) {
    Tensor a({1.0f, 2.0f});

    EXPECT_THROW(Tensor::stack({a}, -1), std::out_of_range);
    EXPECT_THROW(Tensor::stack({a}, 2), std::out_of_range);
}

TEST(TensorStackTest, RejectsFourDimensionalInput) {
    af::array data = af::constant(1.0f, 2, 3, 4, 5);
    Tensor a(data, true);

    EXPECT_THROW(Tensor::stack({a}, 0), std::invalid_argument);
}