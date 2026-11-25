#include "neuro/GradMode.hpp"
#include "neuro/Tensor.hpp"

#include <af/algorithm.h>
#include <af/data.h>
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <vector>

using neuro::Tensor;

namespace {

dim_t offset(const af::dim4& shape, dim_t n, dim_t c, dim_t h, dim_t w) {
    return n + shape[0] * (c + shape[1] * (h + shape[2] * w));
}

std::vector<float> toHost(const af::array& array) {
    std::vector<float> values(array.elements());
    array.as(f32).host(values.data());
    return values;
}

void expectValues(const af::array& array, const std::vector<float>& expected,
                  float tolerance = 1e-5f) {
    ASSERT_EQ(array.elements(), expected.size());

    const auto actual = toHost(array);

    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], tolerance) << "Element " << i;
    }
}

Tensor makeTensor(const af::dim4& shape, bool requires_grad = false) {
    std::vector<float> values(shape.elements());

    // Unique values avoid ambiguous maxima in gradient checks.
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] =
            static_cast<float>((i * 37) % values.size()) / static_cast<float>(values.size()) - 0.5f;
    }

    return Tensor(af::array(shape, values.data()), requires_grad);
}

Tensor constantTensor(const af::dim4& shape, float value, bool requires_grad = false) {
    return Tensor(af::constant(value, shape, f32), requires_grad);
}

// Construct a single-channel image from row-major values.
Tensor image(dim_t height, dim_t width, const std::vector<float>& values,
             bool requires_grad = false) {
    const af::dim4 shape(1, 1, height, width);
    std::vector<float> buffer(shape.elements());

    for (dim_t h = 0; h < height; ++h) {
        for (dim_t w = 0; w < width; ++w) {
            buffer[offset(shape, 0, 0, h, w)] = values[h * width + w];
        }
    }

    return Tensor(af::array(shape, buffer.data()), requires_grad);
}

struct PoolReference {
    af::dim4 shape;
    std::vector<float> values;
    std::vector<dim_t> indices;
};

// Independent CPU implementation of max pooling.
PoolReference referenceMaxPool(const Tensor& input, int window, int stride, int padding) {

    const auto shape = input.shape();
    const auto values = toHost(input.data());

    const dim_t out_h = (shape[2] + 2LL * padding - window) / stride + 1;

    const dim_t out_w = (shape[3] + 2LL * padding - window) / stride + 1;

    const af::dim4 output_shape(shape[0], shape[1], out_h, out_w);

    PoolReference result{output_shape, std::vector<float>(output_shape.elements()),
                         std::vector<dim_t>(output_shape.elements(), -1)};

    for (dim_t n = 0; n < shape[0]; ++n) {
        for (dim_t c = 0; c < shape[1]; ++c) {
            for (dim_t oh = 0; oh < out_h; ++oh) {
                for (dim_t ow = 0; ow < out_w; ++ow) {
                    float maximum = -std::numeric_limits<float>::infinity();

                    dim_t maximum_index = -1;

                    for (int kh = 0; kh < window; ++kh) {
                        for (int kw = 0; kw < window; ++kw) {
                            const dim_t ih = oh * stride - padding + kh;

                            const dim_t iw = ow * stride - padding + kw;

                            if (ih < 0 || ih >= shape[2] || iw < 0 || iw >= shape[3]) {
                                continue;
                            }

                            const dim_t index = offset(shape, n, c, ih, iw);

                            if (values[index] > maximum) {
                                maximum = values[index];
                                maximum_index = index;
                            }
                        }
                    }

                    const dim_t index = offset(output_shape, n, c, oh, ow);

                    result.values[index] = maximum;
                    result.indices[index] = maximum_index;
                }
            }
        }
    }

    return result;
}

void expectReferenceMatch(const af::dim4& shape, int window, int stride, int padding) {

    const Tensor input = makeTensor(shape);

    const auto expected = referenceMaxPool(input, window, stride, padding);

    const Tensor actual = input.maxpool2d(window, stride, padding);

    ASSERT_EQ(actual.shape(), expected.shape);
    expectValues(actual.data(), expected.values);
}

void expectBackwardReferenceMatch(const af::dim4& shape, int window, int stride, int padding) {

    Tensor input = makeTensor(shape, true);

    const auto reference = referenceMaxPool(input, window, stride, padding);

    Tensor output = input.maxpool2d(window, stride, padding);

    std::vector<float> upstream(reference.shape.elements());

    for (std::size_t i = 0; i < upstream.size(); ++i) {
        upstream[i] = static_cast<float>(static_cast<int>(i % 11) - 5) / 4.0f;
    }

    Tensor weights(af::array(reference.shape, upstream.data()));

    (output * weights).sum().backward();

    std::vector<float> expected(shape.elements(), 0.0f);

    for (std::size_t i = 0; i < upstream.size(); ++i) {
        ASSERT_GE(reference.indices[i], 0);
        expected[reference.indices[i]] += upstream[i];
    }

    ASSERT_EQ(input.grad().dims(), shape);
    expectValues(input.grad(), expected);
}

void expectNumericalGradient(const af::dim4& shape, int window, int stride, int padding) {

    Tensor input = makeTensor(shape, true);
    Tensor output = input.maxpool2d(window, stride, padding);

    const auto output_shape = output.shape();

    std::vector<float> upstream(output_shape.elements());

    for (std::size_t i = 0; i < upstream.size(); ++i) {
        upstream[i] = static_cast<float>(static_cast<int>(i % 7) - 3) / 4.0f;
    }

    Tensor weights(af::array(output_shape, upstream.data()));

    (output * weights).sum().backward();

    const auto analytical = toHost(input.grad());
    const auto original = toHost(input.data());

    constexpr float epsilon = 1e-4f;
    constexpr double tolerance = 2e-3;

    neuro::NoGradGuard guard;

    auto evaluate = [&](const std::vector<float>& values) {
        Tensor perturbed(af::array(shape, values.data()));

        const Tensor pooled = perturbed.maxpool2d(window, stride, padding);

        return af::sum<double>((pooled.data() * weights.data()).as(f64));
    };

    for (std::size_t i = 0; i < original.size(); ++i) {
        auto positive = original;
        auto negative = original;

        positive[i] += epsilon;
        negative[i] -= epsilon;

        const double numerical = (evaluate(positive) - evaluate(negative)) /
                                 static_cast<double>(positive[i] - negative[i]);

        EXPECT_NEAR(analytical[i], numerical, tolerance) << "Gradient at element " << i;
    }
}

} // namespace

// ============================================================
// Forward correctness
// ============================================================

TEST(MaxPool2DTest, SingleWindow) {
    const Tensor input = image(2, 2, {1, 4, 3, 2});

    const Tensor output = input.maxpool2d(2, 2, 0);

    EXPECT_EQ(output.shape(), af::dim4(1, 1, 1, 1));
    expectValues(output.data(), {4});
}

TEST(MaxPool2DTest, NonOverlappingWindows) {
    const Tensor input = image(4, 4, {1, 3, 2, 4, 5, 6, 1, 2, 7, 2, 8, 3, 4, 1, 2, 9});

    const Tensor output = input.maxpool2d(2, 2, 0);

    EXPECT_EQ(output.shape(), af::dim4(1, 1, 2, 2));

    // Column-major storage of [[6, 4], [7, 9]].
    expectValues(output.data(), {6, 7, 4, 9});
}

TEST(MaxPool2DTest, OverlappingWindows) {
    const Tensor input = image(3, 3, {1, 2, 3, 4, 9, 6, 7, 8, 5});

    const Tensor output = input.maxpool2d(2, 1, 0);

    EXPECT_EQ(output.shape(), af::dim4(1, 1, 2, 2));
    expectValues(output.data(), {9, 9, 9, 9});
}

TEST(MaxPool2DTest, NegativeInput) {
    const Tensor input = image(2, 2, {-4, -2, -3, -5});

    expectValues(input.maxpool2d(2, 2, 0).data(), {-2});
}

TEST(MaxPool2DTest, NegativeInputWithPadding) {
    const Tensor input = image(2, 2, {-4, -2, -3, -5});

    const Tensor output = input.maxpool2d(2, 1, 1);

    EXPECT_EQ(output.shape(), af::dim4(1, 1, 3, 3));

    expectValues(output.data(), {-4, -3, -3, -2, -2, -3, -2, -2, -5});
}

TEST(MaxPool2DTest, UnitWindowIsIdentity) {
    const Tensor input = makeTensor(af::dim4(2, 3, 4, 5));
    const Tensor output = input.maxpool2d(1, 1, 0);

    EXPECT_EQ(output.shape(), input.shape());
    expectValues(output.data(), toHost(input.data()));
}

TEST(MaxPool2DTest, WindowCoversEntireImage) {
    const Tensor input = image(3, 3, {1, 2, 3, 4, 9, 6, 7, 8, 5});

    const Tensor output = input.maxpool2d(3, 1, 0);

    EXPECT_EQ(output.shape(), af::dim4(1, 1, 1, 1));
    expectValues(output.data(), {9});
}

TEST(MaxPool2DTest, NonSquareInput) {
    expectReferenceMatch(af::dim4(1, 1, 4, 7), 2, 2, 0);
}

TEST(MaxPool2DTest, TallInput) {
    expectReferenceMatch(af::dim4(1, 1, 9, 3), 2, 1, 0);
}

TEST(MaxPool2DTest, MultipleChannels) {
    expectReferenceMatch(af::dim4(1, 4, 5, 7), 2, 2, 0);
}

TEST(MaxPool2DTest, MultipleBatches) {
    expectReferenceMatch(af::dim4(3, 1, 5, 7), 2, 2, 0);
}

TEST(MaxPool2DTest, MultipleBatchesAndChannels) {
    expectReferenceMatch(af::dim4(3, 4, 6, 8), 2, 2, 0);
}

TEST(MaxPool2DTest, OddSpatialDimensions) {
    expectReferenceMatch(af::dim4(2, 3, 7, 9), 2, 2, 0);
}

TEST(MaxPool2DTest, StrideLargerThanWindow) {
    expectReferenceMatch(af::dim4(2, 3, 8, 11), 2, 3, 0);
}

TEST(MaxPool2DTest, ThreeByThreeWindow) {
    expectReferenceMatch(af::dim4(2, 3, 7, 9), 3, 2, 0);
}

TEST(MaxPool2DTest, FourByFourWindow) {
    expectReferenceMatch(af::dim4(2, 3, 9, 11), 4, 2, 0);
}

TEST(MaxPool2DTest, PaddingPreservesSpatialDimensions) {
    const Tensor input = makeTensor(af::dim4(2, 3, 5, 7));

    const Tensor output = input.maxpool2d(3, 1, 1);

    EXPECT_EQ(output.shape(), input.shape());

    expectReferenceMatch(input.shape(), 3, 1, 1);
}

TEST(MaxPool2DTest, PaddingAndStride) {
    expectReferenceMatch(af::dim4(2, 3, 7, 9), 3, 2, 1);
}

TEST(MaxPool2DTest, MaximumAllowedPadding) {
    expectReferenceMatch(af::dim4(2, 3, 7, 9), 4, 2, 2);
}

TEST(MaxPool2DTest, WindowLargerThanUnpaddedInput) {
    expectReferenceMatch(af::dim4(1, 1, 2, 2), 3, 1, 1);
}

TEST(MaxPool2DTest, SinglePixelWithPadding) {
    const Tensor input = constantTensor(af::dim4(1, 1, 1, 1), -3.0f);

    const Tensor output = input.maxpool2d(3, 1, 1);

    EXPECT_EQ(output.shape(), input.shape());
    expectValues(output.data(), {-3});
}

TEST(MaxPool2DTest, EqualMaximumValues) {
    const Tensor input = constantTensor(af::dim4(2, 3, 4, 6), 7.0f);

    const Tensor output = input.maxpool2d(2, 2, 0);

    expectValues(output.data(), std::vector<float>(output.data().elements(), 7.0f));
}

TEST(MaxPool2DTest, MNISTFeatureMapDimensions) {
    expectReferenceMatch(af::dim4(8, 16, 28, 28), 2, 2, 0);
}

TEST(MaxPool2DTest, CIFARFeatureMapDimensions) {
    expectReferenceMatch(af::dim4(8, 32, 32, 32), 2, 2, 0);
}

TEST(MaxPool2DTest, PreservesFloat32) {
    const Tensor input = makeTensor(af::dim4(2, 3, 4, 5));

    EXPECT_EQ(input.maxpool2d(2, 1, 1).data().type(), f32);
}

TEST(MaxPool2DTest, PreservesFloat64) {
    const Tensor input(af::constant(-2.0, af::dim4(2, 3, 4, 5), f64));

    const Tensor output = input.maxpool2d(2, 1, 1);

    EXPECT_EQ(output.data().type(), f64);

    expectValues(output.data(), std::vector<float>(output.data().elements(), -2.0f));
}

TEST(MaxPool2DTest, DoesNotModifyInput) {
    const Tensor input = makeTensor(af::dim4(2, 3, 5, 7));

    const auto original = toHost(input.data());

    input.maxpool2d(3, 2, 1);

    expectValues(input.data(), original);
}

// ============================================================
// Validation
// ============================================================

TEST(MaxPool2DTest, RejectsEmptyInput) {
    EXPECT_THROW(Tensor(af::array()).maxpool2d(2, 2, 0), std::invalid_argument);
}

TEST(MaxPool2DTest, RejectsZeroWindow) {
    const Tensor input = makeTensor(af::dim4(1, 1, 4, 4));

    EXPECT_THROW(input.maxpool2d(0, 1, 0), std::invalid_argument);
}

TEST(MaxPool2DTest, RejectsNegativeWindow) {
    const Tensor input = makeTensor(af::dim4(1, 1, 4, 4));

    EXPECT_THROW(input.maxpool2d(-2, 1, 0), std::invalid_argument);
}

TEST(MaxPool2DTest, RejectsZeroStride) {
    const Tensor input = makeTensor(af::dim4(1, 1, 4, 4));

    EXPECT_THROW(input.maxpool2d(2, 0, 0), std::invalid_argument);
}

TEST(MaxPool2DTest, RejectsNegativeStride) {
    const Tensor input = makeTensor(af::dim4(1, 1, 4, 4));

    EXPECT_THROW(input.maxpool2d(2, -1, 0), std::invalid_argument);
}

TEST(MaxPool2DTest, RejectsNegativePadding) {
    const Tensor input = makeTensor(af::dim4(1, 1, 4, 4));

    EXPECT_THROW(input.maxpool2d(2, 1, -1), std::invalid_argument);
}

TEST(MaxPool2DTest, RejectsExcessivePadding) {
    const Tensor input = makeTensor(af::dim4(1, 1, 4, 4));

    EXPECT_THROW(input.maxpool2d(3, 1, 2), std::invalid_argument);
}

TEST(MaxPool2DTest, RejectsWindowExceedingHeight) {
    const Tensor input = makeTensor(af::dim4(1, 1, 2, 5));

    EXPECT_THROW(input.maxpool2d(3, 1, 0), std::invalid_argument);
}

TEST(MaxPool2DTest, RejectsWindowExceedingWidth) {
    const Tensor input = makeTensor(af::dim4(1, 1, 5, 2));

    EXPECT_THROW(input.maxpool2d(3, 1, 0), std::invalid_argument);
}

TEST(MaxPool2DTest, RejectsWindowExceedingPaddedInput) {
    const Tensor input = makeTensor(af::dim4(1, 1, 2, 2));

    EXPECT_THROW(input.maxpool2d(5, 1, 1), std::invalid_argument);
}

TEST(MaxPool2DTest, RejectsIntegerInput) {
    const Tensor input(af::constant(1, af::dim4(1, 1, 4, 4), s32));

    EXPECT_THROW(input.maxpool2d(2, 2, 0), std::invalid_argument);
}

TEST(MaxPool2DTest, RejectsUnsignedIntegerInput) {
    const Tensor input(af::constant(1, af::dim4(1, 1, 4, 4), u8));

    EXPECT_THROW(input.maxpool2d(2, 2, 0), std::invalid_argument);
}

// ============================================================
// Analytical backward tests
// ============================================================

TEST(MaxPool2DTest, BackwardSingleWindow) {
    Tensor input = image(2, 2, {1, 4, 3, 2}, true);

    input.maxpool2d(2, 2, 0).sum().backward();

    expectValues(input.grad(), {0, 0, 1, 0});
}

TEST(MaxPool2DTest, BackwardNonOverlappingWindows) {
    Tensor input = image(4, 4, {1, 3, 2, 4, 5, 6, 1, 2, 7, 2, 8, 3, 4, 1, 2, 9}, true);

    input.maxpool2d(2, 2, 0).sum().backward();

    expectValues(input.grad(), {0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1});
}

TEST(MaxPool2DTest, BackwardOverlappingWindows) {
    Tensor input = image(3, 3, {1, 2, 3, 4, 9, 6, 7, 8, 5}, true);

    input.maxpool2d(2, 1, 0).sum().backward();

    expectValues(input.grad(), {0, 0, 0, 0, 4, 0, 0, 0, 0});
}

TEST(MaxPool2DTest, BackwardWeightedOverlappingWindows) {
    Tensor input = image(3, 3, {1, 2, 3, 4, 9, 6, 7, 8, 5}, true);

    const Tensor weights = image(2, 2, {1, 2, 3, 4});

    (input.maxpool2d(2, 1, 0) * weights).sum().backward();

    expectValues(input.grad(), {0, 0, 0, 0, 10, 0, 0, 0, 0});
}

TEST(MaxPool2DTest, BackwardNegativeInputWithPadding) {
    Tensor input = image(2, 2, {-4, -2, -3, -5}, true);

    input.maxpool2d(2, 1, 1).sum().backward();

    EXPECT_EQ(input.grad().dims(), input.shape());

    expectValues(input.grad(), {1, 3, 4, 1});
}

TEST(MaxPool2DTest, BackwardUnitWindow) {
    Tensor input = makeTensor(af::dim4(2, 3, 4, 5), true);

    input.maxpool2d(1, 1, 0).sum().backward();

    expectValues(input.grad(), std::vector<float>(input.data().elements(), 1.0f));
}

TEST(MaxPool2DTest, BackwardUnusedPixelsReceiveZero) {
    Tensor input = image(3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9}, true);

    input.maxpool2d(2, 2, 0).sum().backward();

    expectValues(input.grad(), {0, 0, 0, 0, 1, 0, 0, 0, 0});
}

TEST(MaxPool2DTest, BackwardStrideLargerThanWindow) {
    Tensor input = image(5, 5, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                                14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
                         true);

    input.maxpool2d(2, 3, 0).sum().backward();

    expectValues(input.grad(),
                 {0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1});
}

TEST(MaxPool2DTest, BackwardEqualMaximaSelectsOneElement) {
    Tensor input = constantTensor(af::dim4(1, 1, 2, 2), 3.0f, true);

    input.maxpool2d(2, 2, 0).sum().backward();

    const auto gradient = toHost(input.grad());

    // Do not assume a particular tie-breaking index.
    EXPECT_EQ(std::count(gradient.begin(), gradient.end(), 1.0f), 1);
    EXPECT_EQ(std::count(gradient.begin(), gradient.end(), 0.0f), 3);
}

TEST(MaxPool2DTest, BackwardSinglePixelWithPadding) {
    Tensor input = constantTensor(af::dim4(1, 1, 1, 1), -3.0f, true);

    input.maxpool2d(3, 1, 1).sum().backward();

    expectValues(input.grad(), {1});
}

TEST(MaxPool2DTest, BackwardFloat64) {
    Tensor input(af::constant(2.0, af::dim4(1, 1, 2, 2), f64), true);

    input.maxpool2d(2, 2, 0).sum().backward();

    EXPECT_EQ(input.grad().type(), f64);
    EXPECT_NEAR(af::sum<double>(input.grad()), 1.0, 1e-10);
}

// ============================================================
// Backward reference comparisons
// ============================================================

TEST(MaxPool2DTest, BackwardReferenceNonSquare) {
    expectBackwardReferenceMatch(af::dim4(1, 1, 5, 7), 2, 2, 0);
}

TEST(MaxPool2DTest, BackwardReferenceMultipleChannels) {
    expectBackwardReferenceMatch(af::dim4(1, 3, 5, 7), 2, 1, 0);
}

TEST(MaxPool2DTest, BackwardReferenceMultipleBatches) {
    expectBackwardReferenceMatch(af::dim4(2, 1, 5, 7), 2, 2, 0);
}

TEST(MaxPool2DTest, BackwardReferenceBatchesAndChannels) {
    expectBackwardReferenceMatch(af::dim4(2, 3, 5, 7), 2, 2, 0);
}

TEST(MaxPool2DTest, BackwardReferenceOverlappingWindows) {
    expectBackwardReferenceMatch(af::dim4(2, 3, 5, 7), 3, 1, 0);
}

TEST(MaxPool2DTest, BackwardReferencePadding) {
    expectBackwardReferenceMatch(af::dim4(2, 3, 5, 7), 3, 1, 1);
}

TEST(MaxPool2DTest, BackwardReferencePaddingAndStride) {
    expectBackwardReferenceMatch(af::dim4(2, 3, 5, 7), 3, 2, 1);
}

TEST(MaxPool2DTest, BackwardReferenceMaximumPadding) {
    expectBackwardReferenceMatch(af::dim4(2, 3, 5, 7), 4, 2, 2);
}

TEST(MaxPool2DTest, BackwardReferenceOddDimensions) {
    expectBackwardReferenceMatch(af::dim4(2, 3, 7, 9), 2, 2, 0);
}

TEST(MaxPool2DTest, BackwardReferenceLargeStride) {
    expectBackwardReferenceMatch(af::dim4(2, 3, 7, 9), 2, 4, 0);
}

// ============================================================
// Numerical gradient checks
// ============================================================

TEST(MaxPool2DTest, NumericalGradientSingleWindow) {
    expectNumericalGradient(af::dim4(1, 1, 2, 2), 2, 2, 0);
}

TEST(MaxPool2DTest, NumericalGradientNonOverlapping) {
    expectNumericalGradient(af::dim4(1, 1, 4, 4), 2, 2, 0);
}

TEST(MaxPool2DTest, NumericalGradientOverlapping) {
    expectNumericalGradient(af::dim4(1, 1, 4, 4), 2, 1, 0);
}

TEST(MaxPool2DTest, NumericalGradientNonSquare) {
    expectNumericalGradient(af::dim4(1, 1, 3, 5), 2, 1, 0);
}

TEST(MaxPool2DTest, NumericalGradientPadding) {
    expectNumericalGradient(af::dim4(1, 1, 3, 4), 3, 1, 1);
}

TEST(MaxPool2DTest, NumericalGradientPaddingAndStride) {
    expectNumericalGradient(af::dim4(1, 1, 4, 5), 3, 2, 1);
}

TEST(MaxPool2DTest, NumericalGradientMultipleChannels) {
    expectNumericalGradient(af::dim4(1, 2, 3, 4), 2, 1, 0);
}

TEST(MaxPool2DTest, NumericalGradientMultipleBatches) {
    expectNumericalGradient(af::dim4(2, 1, 3, 4), 2, 2, 0);
}

TEST(MaxPool2DTest, NumericalGradientBatchesAndChannels) {
    expectNumericalGradient(af::dim4(2, 2, 3, 4), 2, 1, 1);
}

TEST(MaxPool2DTest, NumericalGradientLargeStride) {
    expectNumericalGradient(af::dim4(1, 1, 5, 5), 2, 3, 0);
}

// ============================================================
// Autograd integration
// ============================================================

TEST(MaxPool2DTest, OutputRequiresGrad) {
    const Tensor input = makeTensor(af::dim4(1, 1, 4, 4), true);

    EXPECT_TRUE(input.maxpool2d(2, 2, 0).requires_grad());
}

TEST(MaxPool2DTest, NonTrainableInputProducesDetachedOutput) {
    const Tensor input = makeTensor(af::dim4(1, 1, 4, 4));

    EXPECT_FALSE(input.maxpool2d(2, 2, 0).requires_grad());
}

TEST(MaxPool2DTest, RespectsNoGradGuard) {
    const Tensor input = makeTensor(af::dim4(1, 1, 4, 4), true);

    neuro::NoGradGuard guard;

    EXPECT_FALSE(input.maxpool2d(2, 2, 0).requires_grad());
}

TEST(MaxPool2DTest, BackwardScaledOutput) {
    Tensor input = image(2, 2, {1, 2, 3, 4}, true);

    (input.maxpool2d(2, 2, 0) * 5.0f).sum().backward();

    expectValues(input.grad(), {0, 0, 0, 5});
}

TEST(MaxPool2DTest, BackwardThroughInputTransformation) {
    Tensor input = image(2, 2, {1, 2, 3, 4}, true);

    (input * 3.0f).maxpool2d(2, 2, 0).sum().backward();

    expectValues(input.grad(), {0, 0, 0, 3});
}

TEST(MaxPool2DTest, BackwardSharedInput) {
    Tensor input = image(2, 2, {1, 2, 3, 4}, true);

    Tensor first = input.maxpool2d(2, 2, 0);
    Tensor second = input.maxpool2d(2, 1, 0);

    (first + second).sum().backward();

    expectValues(input.grad(), {0, 0, 0, 2});
}

TEST(MaxPool2DTest, BackwardRepeatedOutput) {
    Tensor input = image(2, 2, {1, 2, 3, 4}, true);

    Tensor pooled = input.maxpool2d(2, 2, 0);

    (pooled + pooled).sum().backward();

    expectValues(input.grad(), {0, 0, 0, 2});
}

TEST(MaxPool2DTest, BackwardStackedPooling) {
    Tensor input = image(4, 4, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, true);

    Tensor output = input.maxpool2d(2, 2, 0).maxpool2d(2, 2, 0);

    expectValues(output.data(), {16});

    output.sum().backward();

    std::vector<float> expected(16, 0.0f);
    expected[15] = 1.0f;

    expectValues(input.grad(), expected);
}

TEST(MaxPool2DTest, BackwardThroughFlatten) {
    Tensor input = image(4, 4, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, true);

    Tensor output = input.maxpool2d(2, 2, 0).flatten();

    EXPECT_EQ(output.shape(), af::dim4(1, 4));

    output.sum().backward();

    expectValues(input.grad(), {0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1});
}

TEST(MaxPool2DTest, BackwardThroughConvolution) {
    Tensor input = image(3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9}, true);

    Tensor weights = constantTensor(af::dim4(1, 1, 2, 2), 1.0f, true);

    Tensor output = input.conv2d(weights, 1, 0, 1).maxpool2d(2, 2, 0);

    expectValues(output.data(), {28});

    output.sum().backward();

    expectValues(input.grad(), {0, 0, 0, 0, 1, 1, 0, 1, 1});

    expectValues(weights.grad(), {5, 8, 6, 9});
}

TEST(MaxPool2DTest, ZeroGradClearsPoolingGradient) {
    Tensor input = makeTensor(af::dim4(1, 1, 4, 4), true);

    input.maxpool2d(2, 2, 0).sum().backward();

    EXPECT_NEAR(af::sum<double>(input.grad()), 4.0, 1e-5);

    input.zeroGrad();

    expectValues(input.grad(), std::vector<float>(16, 0.0f));
}

TEST(MaxPool2DTest, BackwardDoesNotModifyInput) {
    Tensor input = makeTensor(af::dim4(2, 3, 5, 7), true);

    const auto original = toHost(input.data());

    input.maxpool2d(3, 1, 1).sum().backward();

    expectValues(input.data(), original);
}