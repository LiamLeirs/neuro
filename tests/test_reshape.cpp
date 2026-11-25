
#include "neuro/GradMode.hpp"
#include "neuro/Tensor.hpp"

#include <af/data.h>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

using namespace neuro;

namespace {

void expectValues(const af::array& array, const std::vector<float>& expected) {
    ASSERT_EQ(array.elements(), expected.size());

    std::vector<float> actual(array.elements());
    array.as(f32).host(actual.data());

    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], 1e-5f) << "Mismatch at linear index " << i;
    }
}

} // namespace

// ========================================
// Forward pass
// ========================================

TEST(ReshapeTest, Reshape1DTo2D) {
    Tensor x({1, 2, 3, 4, 5, 6});

    Tensor y = x.reshape(af::dim4(2, 3));

    EXPECT_EQ(y.shape(), af::dim4(2, 3));
    expectValues(y.data(), {1, 2, 3, 4, 5, 6});
}

TEST(ReshapeTest, Reshape2DTo1D) {
    Tensor x({{1, 2, 3}, {4, 5, 6}});

    Tensor y = x.reshape(af::dim4(6));

    EXPECT_EQ(y.shape(), af::dim4(6));
    expectValues(y.data(), {1, 4, 2, 5, 3, 6});
}

TEST(ReshapeTest, Reshape2DTo2D) {
    Tensor x({{1, 2, 3}, {4, 5, 6}});

    Tensor y = x.reshape(af::dim4(3, 2));

    EXPECT_EQ(y.shape(), af::dim4(3, 2));
    expectValues(y.data(), {1, 4, 2, 5, 3, 6});
}

TEST(ReshapeTest, Reshape1DTo4D) {
    Tensor x({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});

    Tensor y = x.reshape(af::dim4(2, 2, 2, 2));

    EXPECT_EQ(y.shape(), af::dim4(2, 2, 2, 2));

    expectValues(y.data(), {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
}

TEST(ReshapeTest, Reshape4DTo2D) {
    Tensor x(af::iota(af::dim4(2, 2, 2, 2)));

    Tensor y = x.reshape(af::dim4(4, 4));

    EXPECT_EQ(y.shape(), af::dim4(4, 4));

    expectValues(y.data(), {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
}

TEST(ReshapeTest, SameShape) {
    Tensor x({{1, 2}, {3, 4}});

    Tensor y = x.reshape(x.shape());

    EXPECT_EQ(y.shape(), x.shape());
    expectValues(y.data(), {1, 3, 2, 4});
}

TEST(ReshapeTest, SingletonDimensions) {
    Tensor x({1, 2, 3, 4});

    Tensor y = x.reshape(af::dim4(1, 4, 1, 1));

    EXPECT_EQ(y.shape(), af::dim4(1, 4, 1, 1));
    expectValues(y.data(), {1, 2, 3, 4});
}

TEST(ReshapeTest, PreservesDataType) {
    Tensor x(af::constant(7, af::dim4(6), s32));

    Tensor y = x.reshape(af::dim4(2, 3));

    EXPECT_EQ(y.data().type(), s32);
    expectValues(y.data(), {7, 7, 7, 7, 7, 7});
}

TEST(ReshapeTest, DoesNotModifyOriginalTensor) {
    Tensor x({{1, 2, 3}, {4, 5, 6}});

    Tensor y = x.reshape(af::dim4(6));

    EXPECT_EQ(x.shape(), af::dim4(2, 3));
    EXPECT_EQ(y.shape(), af::dim4(6));

    expectValues(x.data(), {1, 4, 2, 5, 3, 6});
}

// ========================================
// Input validation
// ========================================

TEST(ReshapeTest, RejectsLargerElementCount) {
    Tensor x({1, 2, 3, 4, 5, 6});

    EXPECT_THROW(x.reshape(af::dim4(3, 3)), std::invalid_argument);
}

TEST(ReshapeTest, RejectsSmallerElementCount) {
    Tensor x({1, 2, 3, 4, 5, 6});

    EXPECT_THROW(x.reshape(af::dim4(2, 2)), std::invalid_argument);
}

TEST(ReshapeTest, RejectsZeroDimension) {
    Tensor x({1, 2, 3, 4, 5, 6});

    EXPECT_THROW(x.reshape(af::dim4(0, 6)), std::invalid_argument);
}

TEST(ReshapeTest, RejectsNegativeDimension) {
    Tensor x({1, 2, 3, 4, 5, 6});

    EXPECT_THROW(x.reshape(af::dim4(-1, 6)), std::invalid_argument);
}

TEST(ReshapeTest, RejectsMultipleNegativeDimensions) {
    Tensor x({1, 2, 3, 4, 5, 6});

    // The product is six, but the dimensions are invalid.
    EXPECT_THROW(x.reshape(af::dim4(-2, -3)), std::invalid_argument);
}

// ========================================
// Automatic differentiation
// ========================================

TEST(ReshapeTest, PreservesRequiresGrad) {
    Tensor x({1, 2, 3, 4}, true);

    Tensor y = x.reshape(af::dim4(2, 2));

    EXPECT_TRUE(y.requires_grad());
}

TEST(ReshapeTest, BackwardRestoresOriginalShape) {
    Tensor x({{1, 2, 3}, {4, 5, 6}}, true);

    Tensor y = x.reshape(af::dim4(6));

    y.sum().backward();

    EXPECT_EQ(x.grad().dims(), x.shape());
    expectValues(x.grad(), {1, 1, 1, 1, 1, 1});
}

TEST(ReshapeTest, BackwardPreservesGradientOrder) {
    Tensor x({{1, 2, 3}, {4, 5, 6}}, true);

    Tensor y = x.reshape(af::dim4(3, 2));

    Tensor weights({{10, 40}, {20, 50}, {30, 60}});

    Tensor loss = (y * weights).sum();
    loss.backward();

    EXPECT_EQ(x.grad().dims(), af::dim4(2, 3));

    expectValues(x.grad(), {10, 20, 30, 40, 50, 60});
}

TEST(ReshapeTest, BackwardThrough4DReshape) {
    Tensor x({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, true);

    Tensor y = x.reshape(af::dim4(2, 2, 2, 2));

    Tensor weights(af::iota(af::dim4(2, 2, 2, 2)) + 1);
    Tensor loss = (y * weights).sum();
    loss.backward();

    EXPECT_EQ(x.grad().dims(), af::dim4(16));

    expectValues(x.grad(), {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
}

TEST(ReshapeTest, BackwardThroughChainedReshapes) {
    Tensor x({1, 2, 3, 4, 5, 6}, true);

    Tensor y = x.reshape(af::dim4(2, 3));
    Tensor z = y.reshape(af::dim4(3, 2));
    Tensor w = z.reshape(af::dim4(6));

    Tensor weights({2, 4, 6, 8, 10, 12});

    Tensor loss = (w * weights).sum();
    loss.backward();

    EXPECT_EQ(x.grad().dims(), af::dim4(6));
    expectValues(x.grad(), {2, 4, 6, 8, 10, 12});
}

TEST(ReshapeTest, BackwardThroughMultipleBranches) {
    Tensor x({1, 2, 3, 4, 5, 6}, true);

    Tensor a = x.reshape(af::dim4(2, 3));
    Tensor b = x.reshape(af::dim4(3, 2));

    Tensor loss = (a * 2.0f).sum() + (b * 3.0f).sum();

    loss.backward();

    expectValues(x.grad(), {5, 5, 5, 5, 5, 5});
}

TEST(ReshapeTest, BackwardWithRepeatedParent) {
    Tensor x({1, 2, 3, 4}, true);

    Tensor y = x.reshape(af::dim4(2, 2));

    Tensor loss = (y * y).sum();
    loss.backward();

    expectValues(x.grad(), {2, 4, 6, 8});
}

TEST(ReshapeTest, BackwardWithSameShape) {
    Tensor x({1, 2, 3, 4}, true);

    Tensor y = x.reshape(x.shape());

    Tensor loss = (y * 3.0f).sum();
    loss.backward();

    expectValues(x.grad(), {3, 3, 3, 3});
}

TEST(ReshapeTest, BackwardThroughPreviousOperation) {
    Tensor x({1, 2, 3, 4}, true);

    Tensor y = x * x;
    Tensor z = y.reshape(af::dim4(2, 2));

    Tensor loss = z.sum();
    loss.backward();

    expectValues(x.grad(), {2, 4, 6, 8});
}

// ========================================
// Gradient tracking
// ========================================

TEST(ReshapeTest, DetachedTensorRemainsDetached) {
    Tensor x({1, 2, 3, 4}, false);

    Tensor y = x.reshape(af::dim4(2, 2));

    EXPECT_FALSE(y.requires_grad());
}

TEST(ReshapeTest, RespectsNoGradGuard) {
    Tensor x({1, 2, 3, 4}, true);

    NoGradGuard guard;

    Tensor y = x.reshape(af::dim4(2, 2));

    EXPECT_FALSE(y.requires_grad());
    EXPECT_TRUE(x.requires_grad());

    EXPECT_EQ(y.shape(), af::dim4(2, 2));
    expectValues(y.data(), {1, 2, 3, 4});
}

TEST(ReshapeTest, GradientTrackingRestoredAfterGuard) {
    Tensor x({1, 2, 3, 4}, true);

    {
        NoGradGuard guard;

        Tensor y = x.reshape(af::dim4(2, 2));

        EXPECT_FALSE(y.requires_grad());
    }

    Tensor z = x.reshape(af::dim4(2, 2));

    EXPECT_TRUE(z.requires_grad());

    z.sum().backward();

    expectValues(x.grad(), {1, 1, 1, 1});
}
