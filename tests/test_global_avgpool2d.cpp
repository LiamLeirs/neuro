
#include "neuro/GradMode.hpp"
#include "neuro/Tensor.hpp"

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

using neuro::Tensor;

namespace {

std::vector<float> toHost(const af::array& array) {
    std::vector<float> values(array.elements());
    array.as(f32).host(values.data());
    return values;
}

void expectValues(const af::array& actual, const std::vector<float>& expected,
                  float tolerance = 1e-5f) {
    ASSERT_EQ(actual.elements(), expected.size());

    const auto values = toHost(actual);

    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(values[i], expected[i], tolerance) << "Mismatch at index " << i;
    }
}

Tensor constantTensor(const af::dim4& shape, float value, bool requires_grad = false) {
    return Tensor(af::constant(value, shape, f32), requires_grad);
}

Tensor makeTensor(const af::dim4& shape, bool requires_grad = false) {
    std::vector<float> values(shape.elements());

    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(i + 1);
    }

    return Tensor(af::array(shape, values.data()), requires_grad);
}

} // namespace

// ============================================================
// 1. Forward correctness
// ============================================================

TEST(GlobalAvgPool2DTest, BasicForward) {
    auto x = makeTensor(af::dim4(1, 1, 3, 3));

    auto y = x.globalAvgPool2d();

    EXPECT_EQ(y.shape(), af::dim4(1, 1, 1, 1));
    expectValues(y.data(), {5.0f});
}

TEST(GlobalAvgPool2DTest, RectangularInput) {
    auto x = makeTensor(af::dim4(1, 1, 2, 3));

    auto y = x.globalAvgPool2d();

    EXPECT_EQ(y.shape(), af::dim4(1, 1, 1, 1));
    expectValues(y.data(), {3.5f});
}

TEST(GlobalAvgPool2DTest, MultipleChannels) {
    auto x = makeTensor(af::dim4(1, 3, 2, 2));

    auto y = x.globalAvgPool2d();

    EXPECT_EQ(y.shape(), af::dim4(1, 3, 1, 1));

    expectValues(y.data(), {5.5f, 6.5f, 7.5f});
}

TEST(GlobalAvgPool2DTest, MultipleBatches) {
    auto x = makeTensor(af::dim4(2, 1, 2, 2));

    auto y = x.globalAvgPool2d();

    EXPECT_EQ(y.shape(), af::dim4(2, 1, 1, 1));

    expectValues(y.data(), {4.0f, 5.0f});
}

TEST(GlobalAvgPool2DTest, MultipleBatchesAndChannels) {
    auto x = makeTensor(af::dim4(2, 3, 2, 3));

    auto y = x.globalAvgPool2d();

    EXPECT_EQ(y.shape(), af::dim4(2, 3, 1, 1));

    expectValues(y.data(), {16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f});
}

TEST(GlobalAvgPool2DTest, SinglePixelFeatureMaps) {
    auto x = makeTensor(af::dim4(2, 3, 1, 1));

    auto y = x.globalAvgPool2d();

    EXPECT_EQ(y.shape(), x.shape());
    expectValues(y.data(), toHost(x.data()));
}

TEST(GlobalAvgPool2DTest, SingleSpatialDimension) {
    auto x1 = makeTensor(af::dim4(1, 1, 1, 5));
    auto x2 = makeTensor(af::dim4(1, 1, 5, 1));

    auto y1 = x1.globalAvgPool2d();
    auto y2 = x2.globalAvgPool2d();

    EXPECT_EQ(y1.shape(), af::dim4(1, 1, 1, 1));
    EXPECT_EQ(y2.shape(), af::dim4(1, 1, 1, 1));

    expectValues(y1.data(), {3.0f});
    expectValues(y2.data(), {3.0f});
}

TEST(GlobalAvgPool2DTest, ConstantNegativeInput) {
    auto x = constantTensor(af::dim4(4, 16, 7, 5), -3.5f);

    auto y = x.globalAvgPool2d();

    EXPECT_EQ(y.shape(), af::dim4(4, 16, 1, 1));

    expectValues(y.data(), std::vector<float>(64, -3.5f));
}

// ============================================================
// 2. Backward correctness
// ============================================================

TEST(GlobalAvgPool2DTest, BackwardUniformGradients) {
    auto x = makeTensor(af::dim4(1, 1, 3, 3), true);

    x.globalAvgPool2d().sum().backward();

    EXPECT_EQ(x.grad().dims(), x.shape());

    expectValues(x.grad(), std::vector<float>(9, 1.0f / 9.0f));
}

TEST(GlobalAvgPool2DTest, BackwardRectangularInput) {
    auto x = makeTensor(af::dim4(2, 3, 4, 5), true);

    x.globalAvgPool2d().sum().backward();

    EXPECT_EQ(x.grad().dims(), x.shape());

    expectValues(x.grad(), std::vector<float>(120, 1.0f / 20.0f));
}

TEST(GlobalAvgPool2DTest, BackwardWeightedUpstream) {
    auto x = makeTensor(af::dim4(2, 3, 2, 2), true);

    auto y = x.globalAvgPool2d();

    // Each batch-channel pair receives a distinct gradient.
    const float weights[] = {4, 8, 12, 16, 20, 24};

    Tensor upstream(af::array(af::dim4(2, 3), weights));

    (y * upstream).sum().backward();

    // Each upstream gradient is divided by H * W = 4.
    // Batch and channel dimensions vary fastest in memory.
    expectValues(x.grad(),
                 {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6});
}

TEST(GlobalAvgPool2DTest, BackwardSinglePixelFeatureMaps) {
    auto x = makeTensor(af::dim4(2, 3, 1, 1), true);

    x.globalAvgPool2d().sum().backward();

    expectValues(x.grad(), std::vector<float>(6, 1.0f));
}

TEST(GlobalAvgPool2DTest, BackwardThroughInputMultiplication) {
    auto x = makeTensor(af::dim4(1, 2, 2, 3), true);

    auto y = (x * 3.0f).globalAvgPool2d();

    y.sum().backward();

    expectValues(x.grad(), std::vector<float>(12, 0.5f));
}

TEST(GlobalAvgPool2DTest, BackwardThroughBranchingGraph) {
    auto x = makeTensor(af::dim4(1, 2, 2, 2), true);

    auto a = x.globalAvgPool2d();
    auto b = x.globalAvgPool2d();

    (a + b).sum().backward();

    expectValues(x.grad(), std::vector<float>(8, 0.5f));
}

TEST(GlobalAvgPool2DTest, BackwardThroughConvolution) {
    auto x = makeTensor(af::dim4(1, 1, 3, 3), true);

    auto weights = constantTensor(af::dim4(1, 1, 2, 2), 1.0f, true);

    auto y = x.conv2d(weights, 1, 0, 1).globalAvgPool2d();

    EXPECT_EQ(y.shape(), af::dim4(1, 1, 1, 1));

    expectValues(y.data(), {20.0f});

    y.sum().backward();

    expectValues(x.grad(), {0.25f, 0.5f, 0.25f, 0.5f, 1.0f, 0.5f, 0.25f, 0.5f, 0.25f});

    expectValues(weights.grad(), {3.0f, 4.0f, 6.0f, 7.0f});
}

// ============================================================
// 3. Autograd behavior
// ============================================================

TEST(GlobalAvgPool2DTest, RespectsRequiresGrad) {
    auto trainable = makeTensor(af::dim4(2, 3, 4, 4), true);

    auto detached = makeTensor(af::dim4(2, 3, 4, 4), false);

    EXPECT_TRUE(trainable.globalAvgPool2d().requires_grad());

    EXPECT_FALSE(detached.globalAvgPool2d().requires_grad());
}

TEST(GlobalAvgPool2DTest, RespectsNoGradGuard) {
    auto x = makeTensor(af::dim4(2, 3, 4, 4), true);

    neuro::NoGradGuard guard;

    EXPECT_FALSE(x.globalAvgPool2d().requires_grad());
}

// ============================================================
// 4. Floating-point precision
// ============================================================

TEST(GlobalAvgPool2DTest, PreservesFloat32) {
    auto x = makeTensor(af::dim4(2, 3, 4, 4), true);

    auto y = x.globalAvgPool2d();

    EXPECT_EQ(y.data().type(), f32);

    y.sum().backward();

    EXPECT_EQ(x.grad().type(), f32);
}

TEST(GlobalAvgPool2DTest, PreservesFloat64) {
    const double values[] = {1.000000001, 1.000000002, 1.000000003, 1.000000004};

    Tensor x(af::array(af::dim4(1, 1, 2, 2), values), true);

    auto y = x.globalAvgPool2d();

    EXPECT_EQ(y.data().type(), f64);

    EXPECT_NEAR(y.data().scalar<double>(), 1.0000000025, 1e-12);

    y.sum().backward();

    EXPECT_EQ(x.grad().type(), f64);

    std::vector<double> gradients(4);
    x.grad().host(gradients.data());

    for (double grad : gradients) {
        EXPECT_NEAR(grad, 0.25, 1e-12);
    }
}

// ============================================================
// 5. Input validation
// ============================================================

TEST(GlobalAvgPool2DTest, RejectsEmptyInput) {
    Tensor x(af::array{});

    EXPECT_THROW(x.globalAvgPool2d(), std::invalid_argument);
}

TEST(GlobalAvgPool2DTest, RejectsIntegerInput) {
    Tensor x(af::constant(1, af::dim4(2, 3, 4, 4), s32));

    EXPECT_THROW(x.globalAvgPool2d(), std::invalid_argument);
}

TEST(GlobalAvgPool2DTest, RejectsUnsignedIntegerInput) {
    Tensor x(af::constant(1, af::dim4(2, 3, 4, 4), u8));

    EXPECT_THROW(x.globalAvgPool2d(), std::invalid_argument);
}
