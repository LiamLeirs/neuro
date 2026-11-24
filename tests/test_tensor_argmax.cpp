#include "TestUtils.hpp"
#include "neuro/GradMode.hpp"
#include "neuro/Tensor.hpp"

#include <gtest/gtest.h>
#include <stdexcept>

using namespace neuro;
using neuro::testing::expectArrayNear;

// ==================== Forward ====================

TEST(TensorArgmaxTest, OneDimensional) {
    Tensor x({1, 5, 3, 2});

    Tensor result = x.argmax(0);

    expectArrayNear(result.data(), af::dim4(1), {1});
}

TEST(TensorArgmaxTest, AlongDimensionZero) {
    Tensor x({{1, 8, 3}, {4, 2, 9}});

    Tensor result = x.argmax(0);

    expectArrayNear(result.data(), af::dim4(1, 3), {1, 0, 1});
}

TEST(TensorArgmaxTest, AlongDimensionOne) {
    Tensor x({{1, 8, 3}, {9, 2, 4}});

    Tensor result = x.argmax(1);

    expectArrayNear(result.data(), af::dim4(2, 1), {1, 0});
}

TEST(TensorArgmaxTest, NegativeValues) {
    Tensor x({-5, -2, -8, -3});

    Tensor result = x.argmax(0);

    expectArrayNear(result.data(), af::dim4(1), {1});
}

TEST(TensorArgmaxTest, SingleElement) {
    Tensor x({42});

    Tensor result = x.argmax(0);

    expectArrayNear(result.data(), af::dim4(1), {0});
}

TEST(TensorArgmaxTest, SingletonDimension) {
    Tensor x({1, 2, 3});

    Tensor result = x.argmax(1);

    expectArrayNear(result.data(), af::dim4(3, 1), {0, 0, 0});
}

TEST(TensorArgmaxTest, ThreeDimensional) {
    const float values[] = {1, 8, 3, 4, 5, 2, 9, 0};

    Tensor x(af::array(2, 2, 2, values));

    Tensor result = x.argmax(2);

    expectArrayNear(result.data(), af::dim4(2, 2, 1), {1, 0, 1, 0});
}

TEST(TensorArgmaxTest, FourDimensional) {
    const float values[] = {1, 8, 5, 2};

    Tensor x(af::array(2, 1, 1, 2, values));

    Tensor result = x.argmax(3);

    expectArrayNear(result.data(), af::dim4(2, 1, 1, 1), {1, 0});
}

// ==================== Data Type ====================

TEST(TensorArgmaxTest, ReturnsIntegerIndices) {
    Tensor x({1, 5, 3});

    Tensor result = x.argmax(0);

    EXPECT_TRUE(result.data().isinteger());
    EXPECT_EQ(result.data().type(), u32);
}

// ==================== Autograd ====================

TEST(TensorArgmaxTest, DoesNotTrackGradients) {
    Tensor x({1, 5, 3}, true);

    Tensor result = x.argmax(0);

    EXPECT_FALSE(result.requires_grad());
    EXPECT_TRUE(x.requires_grad());
}

TEST(TensorArgmaxTest, RemainsDetachedInSubsequentOperations) {
    Tensor x({1, 5, 3}, true);

    Tensor result = x.argmax(0);
    Tensor output = result * 2.0f;

    EXPECT_FALSE(result.requires_grad());
    EXPECT_FALSE(output.requires_grad());
}

TEST(TensorArgmaxTest, WorksUnderNoGradGuard) {
    Tensor x({1, 5, 3}, true);

    NoGradGuard guard;

    Tensor result = x.argmax(0);

    EXPECT_FALSE(result.requires_grad());
    expectArrayNear(result.data(), af::dim4(1), {1});
}

// ==================== Validation ====================

TEST(TensorArgmaxTest, RejectsNegativeDimension) {
    Tensor x({1, 2, 3});

    EXPECT_THROW(x.argmax(-1), std::out_of_range);
}

TEST(TensorArgmaxTest, RejectsDimensionAboveThree) {
    Tensor x({1, 2, 3});

    EXPECT_THROW(x.argmax(4), std::out_of_range);
}

TEST(TensorArgmaxTest, DoesNotModifyInput) {
    Tensor x({1, 5, 3}, true);

    x.argmax(0);

    expectArrayNear(x.data(), af::dim4(3), {1, 5, 3});
    EXPECT_TRUE(x.requires_grad());
}