
#include "neuro/GradMode.hpp"
#include "neuro/Tensor.hpp"
#include "neuro/nn/Flatten.hpp"
#include "neuro/nn/Linear.hpp"
#include "neuro/nn/Sequential.hpp"

#include <af/algorithm.h>
#include <af/data.h>
#include <arrayfire.h>
#include <gtest/gtest.h>
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
// Tensor flatten: forward propagation
// ========================================

TEST(FlattenTest, Flatten4D) {
    Tensor x(af::iota(af::dim4(2, 3, 4, 5)));

    Tensor y = x.flatten();

    EXPECT_EQ(y.shape(), af::dim4(2, 60));
    EXPECT_EQ(y.data().elements(), 120);
}

TEST(FlattenTest, Flatten3D) {
    Tensor x(af::iota(af::dim4(2, 3, 4)));

    Tensor y = x.flatten();

    EXPECT_EQ(y.shape(), af::dim4(2, 12));
    EXPECT_EQ(y.data().elements(), 24);
}

TEST(FlattenTest, Flatten2DLeavesShapeUnchanged) {
    Tensor x({{1, 2, 3}, {4, 5, 6}});

    Tensor y = x.flatten();

    EXPECT_EQ(y.shape(), af::dim4(2, 3));
    expectValues(y.data(), {1, 4, 2, 5, 3, 6});
}

TEST(FlattenTest, Flatten1DPreservesBatchDimension) {
    Tensor x({1, 2, 3, 4});

    Tensor y = x.flatten();

    EXPECT_EQ(y.shape(), af::dim4(4, 1));
    expectValues(y.data(), {1, 2, 3, 4});
}

TEST(FlattenTest, FlattenScalar) {
    Tensor x({7});

    Tensor y = x.flatten();

    EXPECT_EQ(y.shape(), af::dim4(1, 1));
    expectValues(y.data(), {7});
}

TEST(FlattenTest, SingleSampleBatch) {
    Tensor x(af::iota(af::dim4(1, 2, 2, 2)));

    Tensor y = x.flatten();

    EXPECT_EQ(y.shape(), af::dim4(1, 8));

    expectValues(y.data(), {0, 1, 2, 3, 4, 5, 6, 7});
}

TEST(FlattenTest, SingleChannelBatch) {
    Tensor x(af::iota(af::dim4(3, 1, 2, 2)));

    Tensor y = x.flatten();

    EXPECT_EQ(y.shape(), af::dim4(3, 4));

    expectValues(y.data(), {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
}

TEST(FlattenTest, PreservesColumnMajorFeatureOrdering) {
    Tensor x(af::iota(af::dim4(2, 2, 2, 2)));

    Tensor y = x.flatten();

    EXPECT_EQ(y.shape(), af::dim4(2, 8));

    expectValues(y.data(), {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
}

TEST(FlattenTest, DoesNotMixSamples) {
    // Two samples with four features each.
    // Sample 0: 10, 20, 30, 40
    // Sample 1: 100, 200, 300, 400
    const float values[] = {10, 100, 20, 200, 30, 300, 40, 400};

    Tensor x(af::array(af::dim4(2, 2, 2, 1), values));

    Tensor y = x.flatten();

    EXPECT_EQ(y.shape(), af::dim4(2, 4));

    af::array sample0 = y.data()(0, af::span);
    af::array sample1 = y.data()(1, af::span);

    expectValues(sample0, {10, 20, 30, 40});
    expectValues(sample1, {100, 200, 300, 400});
}

TEST(FlattenTest, PreservesDataType) {
    Tensor x(af::iota(af::dim4(2, 2, 2, 2)).as(s32));

    Tensor y = x.flatten();

    EXPECT_EQ(y.data().type(), s32);
    EXPECT_EQ(y.shape(), af::dim4(2, 8));
}

TEST(FlattenTest, DoesNotModifyOriginalTensor) {
    Tensor x(af::iota(af::dim4(2, 2, 2, 2)));

    Tensor y = x.flatten();

    EXPECT_EQ(x.shape(), af::dim4(2, 2, 2, 2));
    EXPECT_EQ(y.shape(), af::dim4(2, 8));

    expectValues(x.data(), {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
}

TEST(FlattenTest, FlattenIsIdempotent) {
    Tensor x(af::iota(af::dim4(2, 2, 2, 2)));

    Tensor y = x.flatten();
    Tensor z = y.flatten();

    EXPECT_EQ(z.shape(), y.shape());

    expectValues(z.data(), {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
}

TEST(FlattenTest, MNISTBatchShape) {
    Tensor x(af::constant(0, af::dim4(128, 1, 28, 28)));

    Tensor y = x.flatten();

    EXPECT_EQ(y.shape(), af::dim4(128, 784));
}

TEST(FlattenTest, CIFAR10BatchShape) {
    Tensor x(af::constant(0, af::dim4(32, 3, 32, 32)));

    Tensor y = x.flatten();

    EXPECT_EQ(y.shape(), af::dim4(32, 3072));
}

// ========================================
// Automatic differentiation
// ========================================

TEST(FlattenTest, PreservesRequiresGrad) {
    Tensor x(af::iota(af::dim4(2, 2, 2, 2)), true);

    Tensor y = x.flatten();

    EXPECT_TRUE(y.requires_grad());
}

TEST(FlattenTest, BackwardRestoresOriginalShape) {
    Tensor x(af::iota(af::dim4(2, 3, 4, 5)), true);

    Tensor y = x.flatten();

    y.sum().backward();

    EXPECT_EQ(x.grad().dims(), x.shape());
    expectValues(x.grad(), std::vector<float>(120, 1.0f));
}

TEST(FlattenTest, BackwardPreservesGradientOrdering) {
    Tensor x(af::iota(af::dim4(2, 2, 2, 2)), true);

    Tensor y = x.flatten();

    Tensor weights(af::iota(af::dim4(2, 8)) + 1);

    (y * weights).sum().backward();

    EXPECT_EQ(x.grad().dims(), af::dim4(2, 2, 2, 2));

    expectValues(x.grad(), {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
}

TEST(FlattenTest, BackwardDoesNotMixSamples) {
    Tensor x(af::constant(1, af::dim4(2, 2, 2, 1)), true);

    Tensor y = x.flatten();

    Tensor weights({{1, 2, 3, 4}, {10, 20, 30, 40}});

    (y * weights).sum().backward();

    EXPECT_EQ(x.grad().dims(), x.shape());

    expectValues(x.grad(), {1, 10, 2, 20, 3, 30, 4, 40});
}

TEST(FlattenTest, BackwardThroughPreviousOperation) {
    Tensor x(af::iota(af::dim4(2, 2, 2, 2)) + 1, true);

    Tensor y = (x * x).flatten();

    y.sum().backward();

    EXPECT_EQ(x.grad().dims(), x.shape());

    expectValues(x.grad(), {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32});
}

TEST(FlattenTest, BackwardThroughMultipleBranches) {
    Tensor x(af::iota(af::dim4(2, 2, 2, 2)), true);

    Tensor a = x.flatten();
    Tensor b = x.flatten();

    Tensor loss = (a * 2.0f).sum() + (b * 3.0f).sum();

    loss.backward();

    EXPECT_EQ(x.grad().dims(), x.shape());
    expectValues(x.grad(), std::vector<float>(16, 5.0f));
}

TEST(FlattenTest, BackwardWithRepeatedParent) {
    Tensor x(af::iota(af::dim4(2, 2, 2, 2)) + 1, true);

    Tensor y = x.flatten();

    (y * y).sum().backward();

    expectValues(x.grad(), {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32});
}

TEST(FlattenTest, BackwardThroughRepeatedFlatten) {
    Tensor x(af::iota(af::dim4(2, 2, 2, 2)), true);

    Tensor y = x.flatten().flatten();

    (y * 3.0f).sum().backward();

    EXPECT_EQ(x.grad().dims(), x.shape());
    expectValues(x.grad(), std::vector<float>(16, 3.0f));
}

// ========================================
// Gradient tracking
// ========================================

TEST(FlattenTest, DetachedTensorRemainsDetached) {
    Tensor x(af::iota(af::dim4(2, 2, 2, 2)));

    Tensor y = x.flatten();

    EXPECT_FALSE(y.requires_grad());
}

TEST(FlattenTest, RespectsNoGradGuard) {
    Tensor x(af::iota(af::dim4(2, 2, 2, 2)), true);

    NoGradGuard guard;

    Tensor y = x.flatten();

    EXPECT_FALSE(y.requires_grad());
    EXPECT_TRUE(x.requires_grad());
    EXPECT_EQ(y.shape(), af::dim4(2, 8));
}

TEST(FlattenTest, GradientTrackingRestoredAfterGuard) {
    Tensor x(af::iota(af::dim4(2, 2, 2, 2)), true);

    {
        NoGradGuard guard;

        Tensor y = x.flatten();

        EXPECT_FALSE(y.requires_grad());
    }

    Tensor z = x.flatten();

    EXPECT_TRUE(z.requires_grad());

    z.sum().backward();

    expectValues(x.grad(), std::vector<float>(16, 1.0f));
}

// ========================================
// Flatten module
// ========================================

TEST(FlattenModuleTest, ForwardShape) {
    nn::Flatten flatten;

    Tensor x(af::iota(af::dim4(2, 3, 4, 5)));

    Tensor y = flatten(x);

    EXPECT_EQ(y.shape(), af::dim4(2, 60));
}

TEST(FlattenModuleTest, ForwardValues) {
    nn::Flatten flatten;

    Tensor x(af::iota(af::dim4(2, 2, 2, 2)));

    Tensor y = flatten(x);

    expectValues(y.data(), {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
}

TEST(FlattenModuleTest, HasNoParameters) {
    nn::Flatten flatten;

    EXPECT_TRUE(flatten.parameters().empty());
}

TEST(FlattenModuleTest, SupportsBackward) {
    nn::Flatten flatten;

    Tensor x(af::iota(af::dim4(2, 2, 2, 2)), true);

    Tensor y = flatten(x);

    Tensor weights(af::iota(af::dim4(2, 8)) + 1);

    (y * weights).sum().backward();

    EXPECT_EQ(x.grad().dims(), x.shape());

    expectValues(x.grad(), {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
}

TEST(FlattenModuleTest, RespectsNoGradGuard) {
    nn::Flatten flatten;

    Tensor x(af::iota(af::dim4(2, 2, 2, 2)), true);

    NoGradGuard guard;

    Tensor y = flatten(x);

    EXPECT_FALSE(y.requires_grad());
}

// ========================================
// Sequential integration
// ========================================

TEST(FlattenModuleTest, SequentialWithLinear) {
    nn::Sequential model(nn::Flatten(), nn::Linear(8, 3));

    Tensor x(af::iota(af::dim4(4, 2, 2, 2)));

    Tensor y = model(x);

    EXPECT_EQ(y.shape(), af::dim4(4, 3));
}

TEST(FlattenModuleTest, SequentialBackward) {
    nn::Sequential model(nn::Flatten(), nn::Linear(8, 3));

    Tensor x(af::iota(af::dim4(4, 2, 2, 2)), true);

    Tensor y = model(x);

    y.sum().backward();

    EXPECT_EQ(x.grad().dims(), x.shape());
    EXPECT_TRUE(af::allTrue<bool>(!af::isNaN(x.grad())));
    EXPECT_TRUE(af::allTrue<bool>(!af::isInf(x.grad())));
}

TEST(FlattenModuleTest, SequentialParameterCount) {
    nn::Sequential model(nn::Flatten(), nn::Linear(8, 3));

    // Only Linear contributes parameters:
    // weights and bias.
    EXPECT_EQ(model.parameters().size(), 2);
}
