
#include "neuro/GradMode.hpp"
#include "neuro/Tensor.hpp"

#include <af/algorithm.h>
#include <af/data.h>
#include <cstddef>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

using namespace neuro;

namespace {

// Calculate a linear index using ArrayFire's column-major layout.
std::size_t offset(const af::dim4& shape, dim_t d0, dim_t d1, dim_t d2, dim_t d3) {
    return d0 + shape[0] * (d1 + shape[1] * (d2 + shape[2] * d3));
}

std::vector<float> toHost(const af::array& array) {
    std::vector<float> values(array.elements());
    array.as(f32).host(values.data());
    return values;
}

void expectValues(const af::array& array, const std::vector<float>& expected,
                  float tolerance = 1e-4f) {
    ASSERT_EQ(array.elements(), expected.size());

    const auto actual = toHost(array);

    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], tolerance) << "Mismatch at linear index " << i;
    }
}

// Independent CPU reference implementation.
// Input:   (N, Cin, H, W)
// Weights: (Cout, Cin, KH, KW)
// Output:  (N, Cout, Hout, Wout)
Tensor referenceConv2d(const Tensor& input, const Tensor& weights, int stride = 1, int padding = 0,
                       int dilation = 1) {
    const auto xs = input.shape();
    const auto ws = weights.shape();

    const dim_t N = xs[0];
    const dim_t Cin = xs[1];
    const dim_t H = xs[2];
    const dim_t W = xs[3];

    const dim_t Cout = ws[0];
    const dim_t KH = ws[2];
    const dim_t KW = ws[3];

    const dim_t Hout = (H + 2LL * padding - dilation * (KH - 1) - 1) / stride + 1;

    const dim_t Wout = (W + 2LL * padding - dilation * (KW - 1) - 1) / stride + 1;

    const af::dim4 output_shape(N, Cout, Hout, Wout);

    const auto x = toHost(input.data());
    const auto w = toHost(weights.data());

    std::vector<float> output(output_shape.elements(), 0.0f);

    for (dim_t n = 0; n < N; ++n) {
        for (dim_t co = 0; co < Cout; ++co) {
            for (dim_t oh = 0; oh < Hout; ++oh) {
                for (dim_t ow = 0; ow < Wout; ++ow) {

                    float sum = 0.0f;

                    for (dim_t ci = 0; ci < Cin; ++ci) {
                        for (dim_t kh = 0; kh < KH; ++kh) {
                            for (dim_t kw = 0; kw < KW; ++kw) {

                                const dim_t ih = oh * stride - padding + kh * dilation;

                                const dim_t iw = ow * stride - padding + kw * dilation;

                                if (ih < 0 || ih >= H || iw < 0 || iw >= W) {
                                    continue;
                                }

                                sum += x[offset(xs, n, ci, ih, iw)] * w[offset(ws, co, ci, kh, kw)];
                            }
                        }
                    }

                    output[offset(output_shape, n, co, oh, ow)] = sum;
                }
            }
        }
    }

    return Tensor(af::array(output_shape, output.data()));
}

// Deterministic, non-uniform test data with positive
// and negative values. Avoids relying on random seeds.
Tensor makeTensor(const af::dim4& shape) {
    std::vector<float> values(shape.elements());

    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 8.0f;
    }

    return Tensor(af::array(shape, values.data()));
}

void expectReferenceMatch(const Tensor& input, const Tensor& weights, int stride = 1,
                          int padding = 0, int dilation = 1) {
    Tensor actual = input.conv2d(weights, stride, padding, dilation);

    Tensor expected = referenceConv2d(input, weights, stride, padding, dilation);

    ASSERT_EQ(actual.shape(), expected.shape());

    expectValues(actual.data(), toHost(expected.data()));
}

// ============================================================================
// Numerical gradient helpers
// ============================================================================

double evaluateConv2dLoss(const std::vector<float>& input_values, const af::dim4& input_shape,
                          const std::vector<float>& weight_values, const af::dim4& weight_shape,
                          const Tensor& upstream, int stride, int padding, int dilation) {
    Tensor input(af::array(input_shape, input_values.data()));
    Tensor weights(af::array(weight_shape, weight_values.data()));

    Tensor output = input.conv2d(weights, stride, padding, dilation);

    return af::sum<double>((output.data() * upstream.data()).as(f64));
}

void expectConv2dGradients(const af::dim4& input_shape, const af::dim4& weight_shape,
                           int stride = 1, int padding = 0, int dilation = 1) {
    constexpr float epsilon = 1e-3f;
    constexpr double tolerance = 2e-3;

    Tensor input(makeTensor(input_shape).data(), true);
    Tensor weights(makeTensor(weight_shape).data(), true);

    Tensor output = input.conv2d(weights, stride, padding, dilation);
    Tensor upstream = makeTensor(output.shape());

    (output * upstream).sum().backward();

    ASSERT_EQ(input.grad().dims(), input_shape);
    ASSERT_EQ(weights.grad().dims(), weight_shape);

    auto input_values = toHost(input.data());
    auto weight_values = toHost(weights.data());

    const auto input_grad = toHost(input.grad());
    const auto weight_grad = toHost(weights.grad());

    const auto evaluate = [&]() {
        return evaluateConv2dLoss(input_values, input_shape, weight_values, weight_shape, upstream,
                                  stride, padding, dilation);
    };

    const auto checkGradients = [&](std::vector<float>& values,
                                    const std::vector<float>& analytical, const char* name) {
        ASSERT_EQ(values.size(), analytical.size());

        for (std::size_t i = 0; i < values.size(); ++i) {
            const float original = values[i];

            values[i] = original + epsilon;
            const float positive_value = values[i];
            const double positive_loss = evaluate();

            values[i] = original - epsilon;
            const float negative_value = values[i];
            const double negative_loss = evaluate();

            values[i] = original;

            const double numerical = (positive_loss - negative_loss) /
                                     (static_cast<double>(positive_value) - negative_value);

            EXPECT_NEAR(analytical[i], numerical, tolerance)
                << name << " gradient mismatch at index " << i;
        }
    };

    checkGradients(input_values, input_grad, "Input");
    checkGradients(weight_values, weight_grad, "Weight");
}

} // namespace

// ========================================
// Basic numerical correctness
// ========================================

TEST(Conv2DTest, SingleChannelKnownValues) {
    // Input:
    // 1 2 3
    // 4 5 6
    // 7 8 9
    const float input_values[] = {1, 4, 7, 2, 5, 8, 3, 6, 9};

    Tensor input(af::array(af::dim4(1, 1, 3, 3), input_values));

    Tensor weights(af::constant(1.0f, af::dim4(1, 1, 2, 2)));

    Tensor output = input.conv2d(weights);

    ASSERT_EQ(output.shape(), af::dim4(1, 1, 2, 2));

    // Expected:
    // 12 16
    // 24 28
    expectValues(output.data(), {12, 24, 16, 28});
}

TEST(Conv2DTest, AsymmetricKernelUsesCrossCorrelation) {
    const float input_values[] = {1, 4, 7, 2, 5, 8, 3, 6, 9};

    // Kernel:
    // 1 2
    // 3 4
    const float weight_values[] = {1, 3, 2, 4};

    Tensor input(af::array(af::dim4(1, 1, 3, 3), input_values));

    Tensor weights(af::array(af::dim4(1, 1, 2, 2), weight_values));

    Tensor output = input.conv2d(weights);

    ASSERT_EQ(output.shape(), af::dim4(1, 1, 2, 2));

    // Expected:
    // 37 47
    // 67 77
    expectValues(output.data(), {37, 67, 47, 77});
}

TEST(Conv2DTest, OneByOneKernel) {
    Tensor input = makeTensor(af::dim4(2, 1, 4, 5));

    Tensor weights(af::constant(2.0f, af::dim4(1, 1, 1, 1)));

    Tensor output = input.conv2d(weights);

    ASSERT_EQ(output.shape(), input.shape());

    auto expected = toHost(input.data());

    for (auto& value : expected) {
        value *= 2.0f;
    }

    expectValues(output.data(), expected);
}

TEST(Conv2DTest, ZeroKernelProducesZeroOutput) {
    Tensor input = makeTensor(af::dim4(2, 3, 5, 5));

    Tensor weights(af::constant(0.0f, af::dim4(4, 3, 3, 3)));

    Tensor output = input.conv2d(weights);

    ASSERT_EQ(output.shape(), af::dim4(2, 4, 3, 3));

    expectValues(output.data(), std::vector<float>(output.data().elements(), 0.0f));
}

TEST(Conv2DTest, ZeroInputProducesZeroOutput) {
    Tensor input(af::constant(0.0f, af::dim4(2, 3, 5, 5)));

    Tensor weights = makeTensor(af::dim4(4, 3, 3, 3));

    Tensor output = input.conv2d(weights);

    expectValues(output.data(), std::vector<float>(output.data().elements(), 0.0f));
}

TEST(Conv2DTest, KernelEqualToInputSize) {
    Tensor input(af::constant(1.0f, af::dim4(1, 1, 3, 3)));

    Tensor weights(af::constant(2.0f, af::dim4(1, 1, 3, 3)));

    Tensor output = input.conv2d(weights);

    ASSERT_EQ(output.shape(), af::dim4(1, 1, 1, 1));

    expectValues(output.data(), {18});
}

// ========================================
// Batch and channel correctness
// ========================================

TEST(Conv2DTest, MultipleInputChannels) {
    Tensor input(af::constant(1.0f, af::dim4(1, 3, 4, 4)));

    Tensor weights(af::constant(1.0f, af::dim4(1, 3, 2, 2)));

    Tensor output = input.conv2d(weights);

    ASSERT_EQ(output.shape(), af::dim4(1, 1, 3, 3));

    // 3 channels * 4 kernel elements.
    expectValues(output.data(), std::vector<float>(9, 12.0f));
}

TEST(Conv2DTest, MultipleOutputChannels) {
    Tensor input = makeTensor(af::dim4(1, 2, 5, 5));
    Tensor weights = makeTensor(af::dim4(4, 2, 3, 3));

    expectReferenceMatch(input, weights);
}

TEST(Conv2DTest, MultipleBatchSamples) {
    Tensor input = makeTensor(af::dim4(4, 1, 5, 5));
    Tensor weights = makeTensor(af::dim4(2, 1, 3, 3));

    expectReferenceMatch(input, weights);
}

TEST(Conv2DTest, DoesNotMixBatchSamples) {
    // Sample 0 contains ones.
    // Sample 1 contains tens.
    const float input_values[] = {1, 10, 1, 10, 1, 10, 1, 10};

    Tensor input(af::array(af::dim4(2, 1, 2, 2), input_values));

    Tensor weights(af::constant(1.0f, af::dim4(1, 1, 2, 2)));

    Tensor output = input.conv2d(weights);

    ASSERT_EQ(output.shape(), af::dim4(2, 1, 1, 1));

    expectValues(output.data(), {4, 40});
}

TEST(Conv2DTest, DistinctOutputFilters) {
    Tensor input(af::constant(1.0f, af::dim4(1, 1, 2, 2)));

    // First filter contains ones.
    // Second filter contains threes.
    const float weight_values[] = {1, 3, 1, 3, 1, 3, 1, 3};

    Tensor weights(af::array(af::dim4(2, 1, 2, 2), weight_values));

    Tensor output = input.conv2d(weights);

    ASSERT_EQ(output.shape(), af::dim4(1, 2, 1, 1));

    expectValues(output.data(), {4, 12});
}

TEST(Conv2DTest, MultipleBatchesAndChannels) {
    Tensor input = makeTensor(af::dim4(3, 2, 6, 7));
    Tensor weights = makeTensor(af::dim4(4, 2, 3, 3));

    expectReferenceMatch(input, weights);
}

// ========================================
// Non-square inputs and kernels
// ========================================

TEST(Conv2DTest, NonSquareInput) {
    Tensor input = makeTensor(af::dim4(2, 3, 5, 8));
    Tensor weights = makeTensor(af::dim4(4, 3, 3, 3));

    Tensor output = input.conv2d(weights);

    EXPECT_EQ(output.shape(), af::dim4(2, 4, 3, 6));

    expectReferenceMatch(input, weights);
}

TEST(Conv2DTest, NonSquareKernel) {
    Tensor input = makeTensor(af::dim4(2, 3, 7, 7));
    Tensor weights = makeTensor(af::dim4(4, 3, 2, 4));

    Tensor output = input.conv2d(weights);

    EXPECT_EQ(output.shape(), af::dim4(2, 4, 6, 4));

    expectReferenceMatch(input, weights);
}

TEST(Conv2DTest, NonSquareInputAndKernel) {
    Tensor input = makeTensor(af::dim4(2, 3, 6, 9));
    Tensor weights = makeTensor(af::dim4(4, 3, 2, 3));

    expectReferenceMatch(input, weights);
}

// ========================================
// Stride
// ========================================

TEST(Conv2DTest, StrideTwo) {
    Tensor input = makeTensor(af::dim4(2, 3, 7, 7));
    Tensor weights = makeTensor(af::dim4(4, 3, 3, 3));

    Tensor output = input.conv2d(weights, 2);

    EXPECT_EQ(output.shape(), af::dim4(2, 4, 3, 3));

    expectReferenceMatch(input, weights, 2);
}

TEST(Conv2DTest, StrideThree) {
    Tensor input = makeTensor(af::dim4(2, 3, 9, 10));
    Tensor weights = makeTensor(af::dim4(4, 3, 3, 3));

    expectReferenceMatch(input, weights, 3);
}

TEST(Conv2DTest, StrideWithNonDivisibleDimensions) {
    Tensor input = makeTensor(af::dim4(2, 3, 8, 8));
    Tensor weights = makeTensor(af::dim4(4, 3, 3, 3));

    Tensor output = input.conv2d(weights, 2);

    // Floor division, not rounding up.
    EXPECT_EQ(output.shape(), af::dim4(2, 4, 3, 3));

    expectReferenceMatch(input, weights, 2);
}

TEST(Conv2DTest, StrideLargerThanKernel) {
    Tensor input = makeTensor(af::dim4(2, 3, 9, 9));
    Tensor weights = makeTensor(af::dim4(4, 3, 2, 2));

    expectReferenceMatch(input, weights, 4);
}

// ========================================
// Padding
// ========================================

TEST(Conv2DTest, PaddingOnePreservesSpatialDimensions) {
    Tensor input = makeTensor(af::dim4(2, 3, 5, 7));
    Tensor weights = makeTensor(af::dim4(4, 3, 3, 3));

    Tensor output = input.conv2d(weights, 1, 1);

    EXPECT_EQ(output.shape(), af::dim4(2, 4, 5, 7));

    expectReferenceMatch(input, weights, 1, 1);
}

TEST(Conv2DTest, ZeroPaddingKnownValues) {
    Tensor input(af::constant(1.0f, af::dim4(1, 1, 2, 2)));

    Tensor weights(af::constant(1.0f, af::dim4(1, 1, 3, 3)));

    Tensor output = input.conv2d(weights, 1, 1);

    ASSERT_EQ(output.shape(), af::dim4(1, 1, 2, 2));

    expectValues(output.data(), {4, 4, 4, 4});
}

TEST(Conv2DTest, PaddingLargerThanOne) {
    Tensor input = makeTensor(af::dim4(2, 3, 4, 5));
    Tensor weights = makeTensor(af::dim4(4, 3, 3, 3));

    expectReferenceMatch(input, weights, 1, 2);
}

TEST(Conv2DTest, PaddingAllowsLargerKernel) {
    Tensor input = makeTensor(af::dim4(1, 1, 3, 3));
    Tensor weights = makeTensor(af::dim4(1, 1, 5, 5));

    Tensor output = input.conv2d(weights, 1, 1);

    EXPECT_EQ(output.shape(), af::dim4(1, 1, 1, 1));

    expectReferenceMatch(input, weights, 1, 1);
}

// ========================================
// Dilation
// ========================================

TEST(Conv2DTest, DilationTwo) {
    Tensor input = makeTensor(af::dim4(2, 3, 8, 8));
    Tensor weights = makeTensor(af::dim4(4, 3, 3, 3));

    Tensor output = input.conv2d(weights, 1, 0, 2);

    EXPECT_EQ(output.shape(), af::dim4(2, 4, 4, 4));

    expectReferenceMatch(input, weights, 1, 0, 2);
}

TEST(Conv2DTest, DilationThree) {
    Tensor input = makeTensor(af::dim4(2, 3, 10, 11));
    Tensor weights = makeTensor(af::dim4(4, 3, 3, 3));

    expectReferenceMatch(input, weights, 1, 0, 3);
}

TEST(Conv2DTest, DilationKnownValues) {
    // Input:
    // 1 2 3
    // 4 5 6
    // 7 8 9
    const float input_values[] = {1, 4, 7, 2, 5, 8, 3, 6, 9};

    Tensor input(af::array(af::dim4(1, 1, 3, 3), input_values));

    Tensor weights(af::constant(1.0f, af::dim4(1, 1, 2, 2)));

    Tensor output = input.conv2d(weights, 1, 0, 2);

    ASSERT_EQ(output.shape(), af::dim4(1, 1, 1, 1));

    // Samples the four corners: 1 + 3 + 7 + 9.
    expectValues(output.data(), {20});
}

TEST(Conv2DTest, DilationWithOneByOneKernel) {
    Tensor input = makeTensor(af::dim4(2, 3, 5, 7));
    Tensor weights = makeTensor(af::dim4(4, 3, 1, 1));

    Tensor regular = input.conv2d(weights, 1, 0, 1);
    Tensor dilated = input.conv2d(weights, 1, 0, 3);

    ASSERT_EQ(regular.shape(), dilated.shape());

    expectValues(dilated.data(), toHost(regular.data()));
}

// ========================================
// Combined configurations
// ========================================

TEST(Conv2DTest, StrideAndPadding) {
    Tensor input = makeTensor(af::dim4(2, 3, 8, 8));
    Tensor weights = makeTensor(af::dim4(4, 3, 3, 3));

    Tensor output = input.conv2d(weights, 2, 1);

    EXPECT_EQ(output.shape(), af::dim4(2, 4, 4, 4));

    expectReferenceMatch(input, weights, 2, 1);
}

TEST(Conv2DTest, StrideAndDilation) {
    Tensor input = makeTensor(af::dim4(2, 3, 10, 11));
    Tensor weights = makeTensor(af::dim4(4, 3, 3, 3));

    expectReferenceMatch(input, weights, 2, 0, 2);
}

TEST(Conv2DTest, PaddingAndDilation) {
    Tensor input = makeTensor(af::dim4(2, 3, 8, 8));
    Tensor weights = makeTensor(af::dim4(4, 3, 3, 3));

    Tensor output = input.conv2d(weights, 1, 2, 2);

    EXPECT_EQ(output.shape(), af::dim4(2, 4, 8, 8));

    expectReferenceMatch(input, weights, 1, 2, 2);
}

TEST(Conv2DTest, StridePaddingAndDilation) {
    Tensor input = makeTensor(af::dim4(2, 3, 9, 11));
    Tensor weights = makeTensor(af::dim4(4, 3, 2, 3));

    expectReferenceMatch(input, weights, 2, 2, 2);
}

// ========================================
// Realistic CNN shapes
// ========================================

TEST(Conv2DTest, MNISTFirstLayer) {
    Tensor input(af::constant(1.0f, af::dim4(8, 1, 28, 28)));

    Tensor weights(af::constant(1.0f, af::dim4(16, 1, 3, 3)));

    Tensor output = input.conv2d(weights);

    EXPECT_EQ(output.shape(), af::dim4(8, 16, 26, 26));
}

TEST(Conv2DTest, CIFAR10FirstLayer) {
    Tensor input(af::constant(1.0f, af::dim4(8, 3, 32, 32)));

    Tensor weights(af::constant(1.0f, af::dim4(16, 3, 3, 3)));

    Tensor output = input.conv2d(weights, 1, 1);

    EXPECT_EQ(output.shape(), af::dim4(8, 16, 32, 32));
}

// ========================================
// Tensor integrity
// ========================================

TEST(Conv2DTest, DoesNotModifyInputOrWeights) {
    Tensor input = makeTensor(af::dim4(2, 3, 5, 7));
    Tensor weights = makeTensor(af::dim4(4, 3, 2, 3));

    const auto input_before = toHost(input.data());
    const auto weights_before = toHost(weights.data());

    Tensor output = input.conv2d(weights);

    EXPECT_EQ(input.shape(), af::dim4(2, 3, 5, 7));
    EXPECT_EQ(weights.shape(), af::dim4(4, 3, 2, 3));

    expectValues(input.data(), input_before);
    expectValues(weights.data(), weights_before);
}

TEST(Conv2DTest, DefaultArgumentsMatchExplicitArguments) {
    Tensor input = makeTensor(af::dim4(2, 3, 5, 5));
    Tensor weights = makeTensor(af::dim4(4, 3, 3, 3));

    Tensor a = input.conv2d(weights);
    Tensor b = input.conv2d(weights, 1, 0, 1);

    ASSERT_EQ(a.shape(), b.shape());

    expectValues(a.data(), toHost(b.data()));
}

TEST(Conv2DTest, PreservesFloatDataType) {
    Tensor input = makeTensor(af::dim4(2, 3, 5, 5));
    Tensor weights = makeTensor(af::dim4(4, 3, 3, 3));

    Tensor output = input.conv2d(weights);

    EXPECT_EQ(output.data().type(), f32);
}

// ========================================
// Input validation
// ========================================

TEST(Conv2DTest, RejectsZeroStride) {
    Tensor input = makeTensor(af::dim4(1, 1, 5, 5));
    Tensor weights = makeTensor(af::dim4(1, 1, 3, 3));

    EXPECT_THROW(input.conv2d(weights, 0), std::invalid_argument);
}

TEST(Conv2DTest, RejectsNegativeStride) {
    Tensor input = makeTensor(af::dim4(1, 1, 5, 5));
    Tensor weights = makeTensor(af::dim4(1, 1, 3, 3));

    EXPECT_THROW(input.conv2d(weights, -1), std::invalid_argument);
}

TEST(Conv2DTest, RejectsNegativePadding) {
    Tensor input = makeTensor(af::dim4(1, 1, 5, 5));
    Tensor weights = makeTensor(af::dim4(1, 1, 3, 3));

    EXPECT_THROW(input.conv2d(weights, 1, -1), std::invalid_argument);
}

TEST(Conv2DTest, RejectsZeroDilation) {
    Tensor input = makeTensor(af::dim4(1, 1, 5, 5));
    Tensor weights = makeTensor(af::dim4(1, 1, 3, 3));

    EXPECT_THROW(input.conv2d(weights, 1, 0, 0), std::invalid_argument);
}

TEST(Conv2DTest, RejectsNegativeDilation) {
    Tensor input = makeTensor(af::dim4(1, 1, 5, 5));
    Tensor weights = makeTensor(af::dim4(1, 1, 3, 3));

    EXPECT_THROW(input.conv2d(weights, 1, 0, -1), std::invalid_argument);
}

TEST(Conv2DTest, RejectsMismatchedChannels) {
    Tensor input = makeTensor(af::dim4(2, 3, 5, 5));
    Tensor weights = makeTensor(af::dim4(4, 2, 3, 3));

    EXPECT_THROW(input.conv2d(weights), std::invalid_argument);
}

TEST(Conv2DTest, RejectsKernelTooTall) {
    Tensor input = makeTensor(af::dim4(1, 1, 3, 5));
    Tensor weights = makeTensor(af::dim4(1, 1, 4, 2));

    EXPECT_THROW(input.conv2d(weights), std::invalid_argument);
}

TEST(Conv2DTest, RejectsKernelTooWide) {
    Tensor input = makeTensor(af::dim4(1, 1, 5, 3));
    Tensor weights = makeTensor(af::dim4(1, 1, 2, 4));

    EXPECT_THROW(input.conv2d(weights), std::invalid_argument);
}

TEST(Conv2DTest, RejectsEffectiveKernelTooLarge) {
    Tensor input = makeTensor(af::dim4(1, 1, 5, 5));
    Tensor weights = makeTensor(af::dim4(1, 1, 3, 3));

    // Effective kernel size is 7x7.
    EXPECT_THROW(input.conv2d(weights, 1, 0, 3), std::invalid_argument);
}

TEST(Conv2DTest, RejectsInsufficientPadding) {
    Tensor input = makeTensor(af::dim4(1, 1, 3, 3));
    Tensor weights = makeTensor(af::dim4(1, 1, 7, 7));

    // Padded input is only 5x5.
    EXPECT_THROW(input.conv2d(weights, 1, 1), std::invalid_argument);
}

TEST(Conv2DTest, RejectsEmptyInput) {
    Tensor input{af::array()};
    Tensor weights = makeTensor(af::dim4(1, 1, 3, 3));

    EXPECT_THROW(input.conv2d(weights), std::invalid_argument);
}

TEST(Conv2DTest, RejectsEmptyWeights) {
    Tensor input = makeTensor(af::dim4(1, 1, 5, 5));
    Tensor weights{af::array()};

    EXPECT_THROW(input.conv2d(weights), std::invalid_argument);
}

// ============================================================================
// Backward: analytical gradients
// ============================================================================

TEST(Conv2DTest, BackwardKnownValues) {
    const float input_values[] = {1, 4, 7, 2, 5, 8, 3, 6, 9};

    const float weight_values[] = {1, 3, 2, 4};

    Tensor input(af::array(af::dim4(1, 1, 3, 3), input_values), true);
    Tensor weights(af::array(af::dim4(1, 1, 2, 2), weight_values), true);

    input.conv2d(weights).sum().backward();

    EXPECT_EQ(input.grad().dims(), input.shape());
    EXPECT_EQ(weights.grad().dims(), weights.shape());

    expectValues(input.grad(), {1, 4, 3, 3, 10, 7, 2, 6, 4});

    expectValues(weights.grad(), {12, 24, 16, 28});
}

TEST(Conv2DTest, BackwardOneByOneKernel) {
    Tensor input(af::constant(3, af::dim4(2, 1, 3, 3), f32), true);
    Tensor weights(af::constant(2, af::dim4(1, 1, 1, 1), f32), true);

    input.conv2d(weights).sum().backward();

    expectValues(input.grad(), std::vector<float>(18, 2.0f));
    expectValues(weights.grad(), {54.0f});
}

TEST(Conv2DTest, BackwardZeroInput) {
    Tensor input(af::constant(0, af::dim4(1, 1, 3, 3), f32), true);
    Tensor weights(af::constant(1, af::dim4(1, 1, 2, 2), f32), true);

    input.conv2d(weights).sum().backward();

    expectValues(input.grad(), {1, 2, 1, 2, 4, 2, 1, 2, 1});

    expectValues(weights.grad(), std::vector<float>(4, 0.0f));
}

TEST(Conv2DTest, BackwardZeroWeights) {
    Tensor input(af::constant(1, af::dim4(1, 1, 3, 3), f32), true);
    Tensor weights(af::constant(0, af::dim4(1, 1, 2, 2), f32), true);

    input.conv2d(weights).sum().backward();

    expectValues(input.grad(), std::vector<float>(9, 0.0f));
    expectValues(weights.grad(), std::vector<float>(4, 4.0f));
}

TEST(Conv2DTest, BackwardMultipleBatchSamples) {
    Tensor input(af::constant(1, af::dim4(3, 1, 3, 3), f32), true);
    Tensor weights(af::constant(1, af::dim4(1, 1, 2, 2), f32), true);

    input.conv2d(weights).sum().backward();

    expectValues(weights.grad(), std::vector<float>(4, 12.0f));
}

TEST(Conv2DTest, BackwardMultipleInputChannels) {
    Tensor input(af::constant(1, af::dim4(1, 3, 3, 3), f32), true);
    Tensor weights(af::constant(1, af::dim4(1, 3, 2, 2), f32), true);

    input.conv2d(weights).sum().backward();

    expectValues(weights.grad(), std::vector<float>(12, 4.0f));
}

TEST(Conv2DTest, BackwardMultipleOutputChannels) {
    Tensor input(af::constant(1, af::dim4(1, 1, 3, 3), f32), true);
    Tensor weights(af::constant(1, af::dim4(2, 1, 2, 2), f32), true);

    input.conv2d(weights).sum().backward();

    expectValues(input.grad(), {2, 4, 2, 4, 8, 4, 2, 4, 2});

    expectValues(weights.grad(), std::vector<float>(8, 4.0f));
}

TEST(Conv2DTest, BackwardStrideTwo) {
    Tensor input(af::constant(1, af::dim4(1, 1, 5, 5), f32), true);
    Tensor weights(af::constant(1, af::dim4(1, 1, 2, 2), f32), true);

    input.conv2d(weights, 2).sum().backward();

    expectValues(input.grad(),
                 {1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0});

    expectValues(weights.grad(), std::vector<float>(4, 4.0f));
}

TEST(Conv2DTest, BackwardPaddingOne) {
    Tensor input(af::constant(1, af::dim4(1, 1, 3, 3), f32), true);
    Tensor weights(af::constant(1, af::dim4(1, 1, 3, 3), f32), true);

    input.conv2d(weights, 1, 1).sum().backward();

    const std::vector<float> expected = {4, 6, 4, 6, 9, 6, 4, 6, 4};

    expectValues(input.grad(), expected);
    expectValues(weights.grad(), expected);
}

TEST(Conv2DTest, BackwardDilationTwo) {
    Tensor input(af::constant(1, af::dim4(1, 1, 3, 3), f32), true);
    Tensor weights(af::constant(1, af::dim4(1, 1, 2, 2), f32), true);

    input.conv2d(weights, 1, 0, 2).sum().backward();

    expectValues(input.grad(), {1, 0, 1, 0, 0, 0, 1, 0, 1});

    expectValues(weights.grad(), std::vector<float>(4, 1.0f));
}

// ============================================================================
// Backward: numerical gradient checks
// ============================================================================

TEST(Conv2DTest, GradientCheckSingleChannel) {
    expectConv2dGradients(af::dim4(1, 1, 3, 3), af::dim4(1, 1, 2, 2));
}

TEST(Conv2DTest, GradientCheckMultipleInputChannels) {
    expectConv2dGradients(af::dim4(1, 2, 3, 3), af::dim4(1, 2, 2, 2));
}

TEST(Conv2DTest, GradientCheckMultipleOutputChannels) {
    expectConv2dGradients(af::dim4(1, 1, 3, 3), af::dim4(2, 1, 2, 2));
}

TEST(Conv2DTest, GradientCheckMultipleBatchSamples) {
    expectConv2dGradients(af::dim4(2, 1, 3, 3), af::dim4(1, 1, 2, 2));
}

TEST(Conv2DTest, GradientCheckMultipleBatchesAndChannels) {
    expectConv2dGradients(af::dim4(2, 2, 3, 3), af::dim4(2, 2, 2, 2));
}

TEST(Conv2DTest, GradientCheckNonSquareInput) {
    expectConv2dGradients(af::dim4(1, 1, 3, 5), af::dim4(1, 1, 2, 2));
}

TEST(Conv2DTest, GradientCheckNonSquareKernel) {
    expectConv2dGradients(af::dim4(1, 1, 4, 4), af::dim4(1, 1, 2, 3));
}

TEST(Conv2DTest, GradientCheckOneByOneKernel) {
    expectConv2dGradients(af::dim4(2, 2, 3, 3), af::dim4(2, 2, 1, 1));
}

TEST(Conv2DTest, GradientCheckStrideTwo) {
    expectConv2dGradients(af::dim4(1, 1, 5, 5), af::dim4(1, 1, 2, 2), 2);
}

TEST(Conv2DTest, GradientCheckStrideWithRemainder) {
    expectConv2dGradients(af::dim4(1, 1, 4, 5), af::dim4(1, 1, 3, 2), 2);
}

TEST(Conv2DTest, GradientCheckPadding) {
    expectConv2dGradients(af::dim4(1, 1, 3, 3), af::dim4(1, 1, 3, 3), 1, 1);
}

TEST(Conv2DTest, GradientCheckDilation) {
    expectConv2dGradients(af::dim4(1, 1, 4, 4), af::dim4(1, 1, 2, 2), 1, 0, 2);
}

TEST(Conv2DTest, GradientCheckStrideAndPadding) {
    expectConv2dGradients(af::dim4(1, 2, 4, 4), af::dim4(2, 2, 3, 3), 2, 1);
}

TEST(Conv2DTest, GradientCheckStrideAndDilation) {
    expectConv2dGradients(af::dim4(1, 1, 5, 5), af::dim4(1, 1, 2, 2), 2, 0, 2);
}

TEST(Conv2DTest, GradientCheckPaddingAndDilation) {
    expectConv2dGradients(af::dim4(1, 1, 4, 4), af::dim4(1, 1, 2, 2), 1, 1, 2);
}

TEST(Conv2DTest, GradientCheckStridePaddingAndDilation) {
    expectConv2dGradients(af::dim4(2, 2, 4, 5), af::dim4(2, 2, 2, 3), 2, 1, 2);
}

// ============================================================================
// Backward: autograd integration
// ============================================================================

TEST(Conv2DTest, BackwardInputOnly) {
    Tensor input(af::constant(1, af::dim4(1, 1, 3, 3), f32), true);
    Tensor weights(af::constant(1, af::dim4(1, 1, 2, 2), f32));

    Tensor output = input.conv2d(weights);

    EXPECT_TRUE(output.requires_grad());

    output.sum().backward();

    expectValues(input.grad(), {1, 2, 1, 2, 4, 2, 1, 2, 1});
}

TEST(Conv2DTest, BackwardWeightsOnly) {
    Tensor input(af::constant(1, af::dim4(1, 1, 3, 3), f32));
    Tensor weights(af::constant(1, af::dim4(1, 1, 2, 2), f32), true);

    Tensor output = input.conv2d(weights);

    EXPECT_TRUE(output.requires_grad());

    output.sum().backward();

    expectValues(weights.grad(), std::vector<float>(4, 4.0f));
}

TEST(Conv2DTest, NeitherParentRequiresGrad) {
    Tensor input(af::constant(1, af::dim4(1, 1, 3, 3), f32));
    Tensor weights(af::constant(1, af::dim4(1, 1, 2, 2), f32));

    EXPECT_FALSE(input.conv2d(weights).requires_grad());
}

TEST(Conv2DTest, BothParentsRequireGrad) {
    Tensor input(af::constant(1, af::dim4(1, 1, 3, 3), f32), true);
    Tensor weights(af::constant(1, af::dim4(1, 1, 2, 2), f32), true);

    EXPECT_TRUE(input.conv2d(weights).requires_grad());
}

TEST(Conv2DTest, RespectsNoGradGuard) {
    Tensor input(af::constant(1, af::dim4(1, 1, 3, 3), f32), true);
    Tensor weights(af::constant(1, af::dim4(1, 1, 2, 2), f32), true);

    {
        neuro::NoGradGuard guard;
        EXPECT_FALSE(input.conv2d(weights).requires_grad());
    }

    EXPECT_TRUE(input.conv2d(weights).requires_grad());
}

TEST(Conv2DTest, BackwardScaledOutput) {
    Tensor input(af::constant(1, af::dim4(1, 1, 3, 3), f32), true);
    Tensor weights(af::constant(1, af::dim4(1, 1, 2, 2), f32), true);

    (input.conv2d(weights) * 3.0f).sum().backward();

    expectValues(input.grad(), {3, 6, 3, 6, 12, 6, 3, 6, 3});

    expectValues(weights.grad(), std::vector<float>(4, 12.0f));
}

TEST(Conv2DTest, BackwardAccumulatesSharedInput) {
    Tensor input(af::constant(1, af::dim4(1, 1, 3, 3), f32), true);

    Tensor weights_a(af::constant(1, af::dim4(1, 1, 2, 2), f32));
    Tensor weights_b(af::constant(2, af::dim4(1, 1, 2, 2), f32));

    Tensor output_a = input.conv2d(weights_a);
    Tensor output_b = input.conv2d(weights_b);

    (output_a + output_b).sum().backward();

    expectValues(input.grad(), {3, 6, 3, 6, 12, 6, 3, 6, 3});
}

TEST(Conv2DTest, BackwardAccumulatesSharedWeights) {
    Tensor input_a(af::constant(1, af::dim4(1, 1, 3, 3), f32));
    Tensor input_b(af::constant(2, af::dim4(1, 1, 3, 3), f32));

    Tensor weights(af::constant(1, af::dim4(1, 1, 2, 2), f32), true);

    Tensor output_a = input_a.conv2d(weights);
    Tensor output_b = input_b.conv2d(weights);

    (output_a + output_b).sum().backward();

    expectValues(weights.grad(), std::vector<float>(4, 12.0f));
}

TEST(Conv2DTest, BackwardRepeatedParent) {
    Tensor tensor(af::constant(2, af::dim4(1, 1, 2, 2), f32), true);

    tensor.conv2d(tensor).sum().backward();

    expectValues(tensor.grad(), std::vector<float>(4, 4.0f));
}

TEST(Conv2DTest, BackwardThroughStackedConvolutions) {
    Tensor input(af::constant(1, af::dim4(1, 1, 3, 3), f32), true);

    Tensor weights_a(af::constant(2, af::dim4(1, 1, 2, 2), f32), true);
    Tensor weights_b(af::constant(3, af::dim4(1, 1, 2, 2), f32), true);

    Tensor intermediate = input.conv2d(weights_a);
    Tensor output = intermediate.conv2d(weights_b);

    output.sum().backward();

    expectValues(input.grad(), {6, 12, 6, 12, 24, 12, 6, 12, 6});

    expectValues(weights_a.grad(), std::vector<float>(4, 12.0f));
    expectValues(weights_b.grad(), std::vector<float>(4, 8.0f));
}

TEST(Conv2DTest, BackwardThroughFlatten) {
    Tensor input(af::constant(1, af::dim4(2, 1, 3, 3), f32), true);
    Tensor weights(af::constant(1, af::dim4(2, 1, 2, 2), f32), true);

    Tensor output = input.conv2d(weights).flatten();

    EXPECT_EQ(output.shape(), af::dim4(2, 8));

    output.sum().backward();

    expectValues(weights.grad(), std::vector<float>(8, 8.0f));

    EXPECT_EQ(input.grad().dims(), input.shape());
    EXPECT_NEAR(af::sum<float>(input.grad()), 64.0f, 1e-4f);
}

TEST(Conv2DTest, BackwardThroughInputTransformation) {
    Tensor input(af::constant(1, af::dim4(1, 1, 3, 3), f32), true);
    Tensor weights(af::constant(1, af::dim4(1, 1, 2, 2), f32), true);

    Tensor transformed = input * 2.0f;

    transformed.conv2d(weights).sum().backward();

    expectValues(input.grad(), {2, 4, 2, 4, 8, 4, 2, 4, 2});

    expectValues(weights.grad(), std::vector<float>(4, 8.0f));
}

TEST(Conv2DTest, BackwardDoesNotModifyInputOrWeights) {
    Tensor input(makeTensor(af::dim4(2, 2, 4, 4)).data(), true);
    Tensor weights(makeTensor(af::dim4(2, 2, 2, 2)).data(), true);

    const auto original_input = toHost(input.data());
    const auto original_weights = toHost(weights.data());

    input.conv2d(weights).sum().backward();

    expectValues(input.data(), original_input);
    expectValues(weights.data(), original_weights);
}

TEST(Conv2DTest, BackwardPreservesGradientDataType) {
    Tensor input(af::constant(1, af::dim4(1, 1, 3, 3), f32), true);
    Tensor weights(af::constant(1, af::dim4(1, 1, 2, 2), f32), true);

    input.conv2d(weights).sum().backward();

    EXPECT_EQ(input.grad().type(), f32);
    EXPECT_EQ(weights.grad().type(), f32);
}
