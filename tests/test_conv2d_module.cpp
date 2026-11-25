
#include "neuro/GradMode.hpp"
#include "neuro/Tensor.hpp"
#include "neuro/nn/Conv2D.hpp"
#include "neuro/nn/Flatten.hpp"
#include "neuro/nn/Linear.hpp"
#include "neuro/nn/ReLU.hpp"
#include "neuro/nn/Sequential.hpp"
#include "neuro/optim/SGD.hpp"

#include <af/algorithm.h>
#include <af/data.h>
#include <arrayfire.h>
#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

using neuro::Tensor;
using neuro::nn::Conv2D;

namespace {

void expectValues(const af::array& array, const std::vector<float>& expected,
                  float tolerance = 1e-4f) {
    ASSERT_EQ(array.elements(), expected.size());

    std::vector<float> actual(array.elements());
    array.as(f32).host(actual.data());

    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], tolerance) << "Mismatch at index " << i;
    }
}

void expectConstant(const af::array& array, float expected, float tolerance = 1e-4f) {
    expectValues(array, std::vector<float>(array.elements(), expected), tolerance);
}

Tensor constantTensor(const af::dim4& shape, float value, bool requires_grad = false) {
    return Tensor(af::constant(value, shape, f32), requires_grad);
}

} // namespace

// ============================================================================
// Initialization
// ============================================================================

TEST(Conv2DModuleTest, WeightShape) {
    Conv2D layer(3, 16, 5);

    EXPECT_EQ(layer.weights().shape(), af::dim4(16, 3, 5, 5));
}

TEST(Conv2DModuleTest, BiasShape) {
    Conv2D layer(3, 16, 3);

    EXPECT_EQ(layer.bias().shape(), af::dim4(1, 16, 1, 1));
}

TEST(Conv2DModuleTest, WeightsRequireGrad) {
    Conv2D layer(3, 16, 3);

    EXPECT_TRUE(layer.weights().requires_grad());
}

TEST(Conv2DModuleTest, BiasRequiresGrad) {
    Conv2D layer(3, 16, 3);

    EXPECT_TRUE(layer.bias().requires_grad());
}

TEST(Conv2DModuleTest, BiasInitializedToZero) {
    Conv2D layer(3, 16, 3);

    expectConstant(layer.bias().data(), 0.0f);
}

TEST(Conv2DModuleTest, GlorotInitializationBounds) {
    constexpr int in_channels = 3;
    constexpr int out_channels = 16;
    constexpr int kernel_size = 3;

    Conv2D layer(in_channels, out_channels, kernel_size);

    const double fan_in = in_channels * kernel_size * kernel_size;
    const double fan_out = out_channels * kernel_size * kernel_size;
    const double limit = std::sqrt(6.0 / (fan_in + fan_out));

    const float minimum = af::min<float>(layer.weights().data());
    const float maximum = af::max<float>(layer.weights().data());

    EXPECT_GE(minimum, -limit - 1e-6);
    EXPECT_LE(maximum, limit + 1e-6);

    // Ensure initialization is not constant.
    EXPECT_GT(maximum - minimum, 0.0f);
}

// ============================================================================
// Constructor validation
// ============================================================================

TEST(Conv2DModuleTest, RejectsInvalidInputChannels) {
    EXPECT_THROW(Conv2D(0, 16, 3), std::invalid_argument);
    EXPECT_THROW(Conv2D(-1, 16, 3), std::invalid_argument);
}

TEST(Conv2DModuleTest, RejectsInvalidOutputChannels) {
    EXPECT_THROW(Conv2D(3, 0, 3), std::invalid_argument);
    EXPECT_THROW(Conv2D(3, -1, 3), std::invalid_argument);
}

TEST(Conv2DModuleTest, RejectsInvalidKernelSize) {
    EXPECT_THROW(Conv2D(3, 16, 0), std::invalid_argument);
    EXPECT_THROW(Conv2D(3, 16, -1), std::invalid_argument);
}

TEST(Conv2DModuleTest, RejectsInvalidStride) {
    EXPECT_THROW(Conv2D(3, 16, 3, 0), std::invalid_argument);
    EXPECT_THROW(Conv2D(3, 16, 3, -1), std::invalid_argument);
}

TEST(Conv2DModuleTest, RejectsNegativePadding) {
    EXPECT_THROW(Conv2D(3, 16, 3, 1, -1), std::invalid_argument);
}

TEST(Conv2DModuleTest, RejectsInvalidDilation) {
    EXPECT_THROW(Conv2D(3, 16, 3, 1, 0, 0), std::invalid_argument);
    EXPECT_THROW(Conv2D(3, 16, 3, 1, 0, -1), std::invalid_argument);
}

// ============================================================================
// Parameter registration
// ============================================================================

TEST(Conv2DModuleTest, ParametersWithBias) {
    Conv2D layer(3, 16, 3);

    const auto params = layer.parameters();

    ASSERT_EQ(params.size(), 2u);

    EXPECT_EQ(params[0].shape(), layer.weights().shape());
    EXPECT_EQ(params[1].shape(), layer.bias().shape());

    EXPECT_TRUE(params[0].requires_grad());
    EXPECT_TRUE(params[1].requires_grad());
}

TEST(Conv2DModuleTest, ParametersWithoutBias) {
    Conv2D layer(3, 16, 3, 1, 0, 1, false);

    const auto params = layer.parameters();

    ASSERT_EQ(params.size(), 1u);
    EXPECT_EQ(params[0].shape(), layer.weights().shape());
    EXPECT_TRUE(params[0].requires_grad());
}

TEST(Conv2DModuleTest, BiasAccessorThrowsWhenDisabled) {
    Conv2D layer(3, 16, 3, 1, 0, 1, false);

    EXPECT_THROW(layer.bias(), std::logic_error);
}

TEST(Conv2DModuleTest, ParametersShareGradientStorage) {
    Conv2D layer(1, 1, 2);

    layer.weights() = constantTensor(af::dim4(1, 1, 2, 2), 1.0f, true);

    const auto params = layer.parameters();

    Tensor input = constantTensor(af::dim4(1, 1, 3, 3), 1.0f);

    layer.forward(input).sum().backward();

    expectConstant(params[0].grad(), 4.0f);
    expectConstant(params[1].grad(), 4.0f);
}

// ============================================================================
// Forward propagation
// ============================================================================

TEST(Conv2DModuleTest, ForwardDefaultConfiguration) {
    Conv2D layer(3, 16, 3);

    Tensor input = constantTensor(af::dim4(4, 3, 28, 28), 1.0f);

    Tensor output = layer.forward(input);

    EXPECT_EQ(output.shape(), af::dim4(4, 16, 26, 26));
}

TEST(Conv2DModuleTest, ForwardWithoutBias) {
    Conv2D layer(1, 1, 2, 1, 0, 1, false);

    layer.weights() = constantTensor(af::dim4(1, 1, 2, 2), 2.0f, true);

    Tensor input = constantTensor(af::dim4(1, 1, 3, 3), 1.0f);

    Tensor output = layer.forward(input);

    EXPECT_EQ(output.shape(), af::dim4(1, 1, 2, 2));
    expectConstant(output.data(), 8.0f);
}

TEST(Conv2DModuleTest, ForwardWithBias) {
    Conv2D layer(1, 1, 2);

    layer.weights() = constantTensor(af::dim4(1, 1, 2, 2), 2.0f, true);

    layer.bias() = constantTensor(af::dim4(1, 1), 3.0f, true);

    Tensor input = constantTensor(af::dim4(1, 1, 3, 3), 1.0f);

    expectConstant(layer.forward(input).data(), 11.0f);
}

TEST(Conv2DModuleTest, BiasBroadcastsAcrossBatchAndSpatialDimensions) {
    Conv2D layer(1, 3, 2);

    layer.weights() = constantTensor(af::dim4(3, 1, 2, 2), 0.0f, true);

    const float bias_values[] = {1.0f, 2.0f, 3.0f};

    layer.bias() = Tensor(af::array(af::dim4(1, 3), bias_values), true);

    Tensor input = constantTensor(af::dim4(2, 1, 4, 5), 1.0f);

    Tensor output = layer.forward(input);

    ASSERT_EQ(output.shape(), af::dim4(2, 3, 3, 4));

    // Each channel has its own bias, shared across batches and pixels.
    std::vector<float> expected;

    for (int w = 0; w < 4; ++w) {
        for (int h = 0; h < 3; ++h) {
            for (int c = 0; c < 3; ++c) {
                for (int n = 0; n < 2; ++n) {
                    expected.push_back(bias_values[c]);
                }
            }
        }
    }

    expectValues(output.data(), expected);
}

TEST(Conv2DModuleTest, ForwardUsesConfiguredStridePaddingAndDilation) {
    Conv2D layer(2, 4, 3, 2, 2, 2);

    Tensor input = constantTensor(af::dim4(2, 2, 9, 11), 1.0f);

    Tensor output = layer.forward(input);

    Tensor expected = input.conv2d(layer.weights(), 2, 2, 2) + layer.bias();

    EXPECT_EQ(output.shape(), af::dim4(2, 4, 5, 6));
    EXPECT_EQ(output.shape(), expected.shape());

    EXPECT_NEAR(af::max<float>(af::abs(output.data() - expected.data())), 0.0f, 1e-4f);
}

TEST(Conv2DModuleTest, ForwardRejectsMismatchedInputChannels) {
    Conv2D layer(3, 16, 3);

    Tensor input = constantTensor(af::dim4(2, 1, 28, 28), 1.0f);

    EXPECT_THROW(layer.forward(input), std::invalid_argument);
}

// ============================================================================
// Backward propagation
// ============================================================================

TEST(Conv2DModuleTest, BackwardComputesAllGradients) {
    Conv2D layer(1, 1, 2);

    layer.weights() = constantTensor(af::dim4(1, 1, 2, 2), 2.0f, true);

    Tensor input = constantTensor(af::dim4(1, 1, 3, 3), 1.0f, true);

    layer.forward(input).sum().backward();

    EXPECT_EQ(input.grad().dims(), input.shape());
    EXPECT_EQ(layer.weights().grad().dims(), layer.weights().shape());
    EXPECT_EQ(layer.bias().grad().dims(), layer.bias().shape());

    expectValues(input.grad(), {2, 4, 2, 4, 8, 4, 2, 4, 2});

    expectConstant(layer.weights().grad(), 4.0f);
    expectConstant(layer.bias().grad(), 4.0f);
}

TEST(Conv2DModuleTest, BackwardWithoutBias) {
    Conv2D layer(1, 1, 2, 1, 0, 1, false);

    layer.weights() = constantTensor(af::dim4(1, 1, 2, 2), 1.0f, true);

    Tensor input = constantTensor(af::dim4(1, 1, 3, 3), 1.0f, true);

    layer.forward(input).sum().backward();

    expectValues(input.grad(), {1, 2, 1, 2, 4, 2, 1, 2, 1});

    expectConstant(layer.weights().grad(), 4.0f);
}

TEST(Conv2DModuleTest, BackwardWithNonTrainableInput) {
    Conv2D layer(1, 1, 2);

    Tensor input = constantTensor(af::dim4(1, 1, 3, 3), 1.0f);

    Tensor output = layer.forward(input);

    EXPECT_FALSE(input.requires_grad());
    EXPECT_TRUE(output.requires_grad());

    output.sum().backward();

    expectConstant(layer.weights().grad(), 4.0f);
    expectConstant(layer.bias().grad(), 4.0f);
}

TEST(Conv2DModuleTest, BiasGradientReducesBatchAndSpatialDimensions) {
    Conv2D layer(2, 3, 3);

    Tensor input = constantTensor(af::dim4(4, 2, 5, 6), 1.0f);

    Tensor output = layer.forward(input);

    ASSERT_EQ(output.shape(), af::dim4(4, 3, 3, 4));

    output.sum().backward();

    EXPECT_EQ(layer.bias().grad().dims(), af::dim4(1, 3));

    // Each bias contributes to 4 * 3 * 4 output elements.
    expectValues(layer.bias().grad(), {48.0f, 48.0f, 48.0f});
}

TEST(Conv2DModuleTest, BiasGradientPreservesChannelIndependence) {
    Conv2D layer(1, 3, 2);

    Tensor input = constantTensor(af::dim4(2, 1, 3, 3), 1.0f);

    const float scales[] = {1.0f, 2.0f, 3.0f};

    Tensor upstream(af::array(af::dim4(1, 3), scales));

    (layer.forward(input) * upstream).sum().backward();

    // Each channel has 2 * 2 * 2 = 8 output elements.
    expectValues(layer.bias().grad(), {8.0f, 16.0f, 24.0f});
}

TEST(Conv2DModuleTest, RespectsNoGradGuard) {
    Conv2D layer(1, 4, 3);

    Tensor input = constantTensor(af::dim4(2, 1, 8, 8), 1.0f, true);

    {
        neuro::NoGradGuard guard;
        EXPECT_FALSE(layer.forward(input).requires_grad());
    }

    EXPECT_TRUE(layer.forward(input).requires_grad());
}

// ============================================================================
// Optimizer integration
// ============================================================================

TEST(Conv2DModuleTest, SGDUpdatesWeightsAndBias) {
    Conv2D layer(1, 1, 2);

    layer.weights() = constantTensor(af::dim4(1, 1, 2, 2), 1.0f, true);

    neuro::SGD optimizer(layer.parameters(), 0.1f);

    Tensor input = constantTensor(af::dim4(1, 1, 3, 3), 1.0f);

    optimizer.zeroGrad();

    layer.forward(input).sum().backward();
    optimizer.step();

    // Each parameter has gradient 4.
    expectConstant(layer.weights().data(), 0.6f);
    expectConstant(layer.bias().data(), -0.4f);
}

TEST(Conv2DModuleTest, OptimizerZeroGradClearsAllParameters) {
    Conv2D layer(1, 1, 2);

    neuro::SGD optimizer(layer.parameters(), 0.1f);

    Tensor input = constantTensor(af::dim4(1, 1, 3, 3), 1.0f);

    layer.forward(input).sum().backward();

    expectConstant(layer.weights().grad(), 4.0f);
    expectConstant(layer.bias().grad(), 4.0f);

    optimizer.zeroGrad();

    expectConstant(layer.weights().grad(), 0.0f);
    expectConstant(layer.bias().grad(), 0.0f);
}

// ============================================================================
// Sequential integration
// ============================================================================

TEST(Conv2DModuleTest, SequentialForward) {
    neuro::nn::Sequential model(Conv2D(1, 4, 3, 1, 1), neuro::nn::ReLU(), Conv2D(4, 8, 3, 1, 1),
                                neuro::nn::ReLU(), neuro::nn::Flatten(),
                                neuro::nn::Linear(8 * 8 * 8, 10));

    Tensor input = constantTensor(af::dim4(2, 1, 8, 8), 1.0f);

    Tensor output = model(input);

    EXPECT_EQ(output.shape(), af::dim4(2, 10));
}

TEST(Conv2DModuleTest, SequentialRegistersConvolutionParameters) {
    neuro::nn::Sequential model(Conv2D(1, 4, 3), neuro::nn::ReLU(),
                                Conv2D(4, 8, 3, 1, 0, 1, false));

    const auto params = model.parameters();

    ASSERT_EQ(params.size(), 3u);

    EXPECT_EQ(params[0].shape(), af::dim4(4, 1, 3, 3));
    EXPECT_EQ(params[1].shape(), af::dim4(1, 4));
    EXPECT_EQ(params[2].shape(), af::dim4(8, 4, 3, 3));
}

TEST(Conv2DModuleTest, SequentialBackward) {
    Conv2D first(1, 2, 2);
    Conv2D second(2, 1, 2);

    first.weights() = constantTensor(af::dim4(2, 1, 2, 2), 1.0f, true);

    second.weights() = constantTensor(af::dim4(1, 2, 2, 2), 1.0f, true);

    neuro::nn::Sequential model(std::move(first), neuro::nn::ReLU(), std::move(second),
                                neuro::nn::Flatten());

    Tensor input = constantTensor(af::dim4(1, 1, 3, 3), 1.0f, true);

    Tensor output = model(input);

    ASSERT_EQ(output.shape(), af::dim4(1, 1));
    expectConstant(output.data(), 32.0f);

    output.sum().backward();

    const auto params = model.parameters();

    ASSERT_EQ(params.size(), 4u);

    expectConstant(params[0].grad(), 4.0f);
    expectConstant(params[1].grad(), 4.0f);
    expectConstant(params[2].grad(), 4.0f);
    expectConstant(params[3].grad(), 1.0f);

    expectValues(input.grad(), {2, 4, 2, 4, 8, 4, 2, 4, 2});
}
