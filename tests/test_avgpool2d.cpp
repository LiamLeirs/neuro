
#include "neuro/GradMode.hpp"
#include "neuro/Tensor.hpp"

#include <arrayfire.h>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

using neuro::Tensor;

namespace {

// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------

std::size_t offset(const af::dim4& shape, dim_t n, dim_t c, dim_t h, dim_t w) {
    return n + shape[0] * (c + shape[1] * (h + shape[2] * w));
}

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
    const auto size = shape.elements();
    std::vector<float> values(size);

    for (dim_t i = 0; i < size; ++i) {
        values[i] = static_cast<float>((i * 37) % 101) / 50.0f - 1.0f;
    }

    return Tensor(af::array(shape, values.data()), requires_grad);
}

// Construct a single-channel image from row-major values.
Tensor image(dim_t height, dim_t width, const std::vector<float>& values,
             bool requires_grad = false) {
    const af::dim4 shape(1, 1, height, width);
    std::vector<float> storage(height * width);

    for (dim_t h = 0; h < height; ++h) {
        for (dim_t w = 0; w < width; ++w) {
            storage[offset(shape, 0, 0, h, w)] = values[h * width + w];
        }
    }

    return Tensor(af::array(shape, storage.data()), requires_grad);
}

// ------------------------------------------------------------
// Independent CPU reference implementation
// ------------------------------------------------------------

struct PoolReference {
    af::dim4 output_shape;
    std::vector<float> output;
    std::vector<float> input_grad;
};

PoolReference referenceAvgPool(const af::dim4& shape, const std::vector<float>& input, int window,
                               int stride, int padding, bool count_include_pad,
                               const std::vector<float>& upstream = {}) {

    const dim_t out_h = (shape[2] + 2 * padding - window) / stride + 1;

    const dim_t out_w = (shape[3] + 2 * padding - window) / stride + 1;

    const af::dim4 output_shape(shape[0], shape[1], out_h, out_w);

    PoolReference result{output_shape, std::vector<float>(output_shape.elements(), 0.0f),
                         std::vector<float>(shape.elements(), 0.0f)};

    for (dim_t n = 0; n < shape[0]; ++n) {
        for (dim_t c = 0; c < shape[1]; ++c) {
            for (dim_t oh = 0; oh < out_h; ++oh) {
                for (dim_t ow = 0; ow < out_w; ++ow) {

                    float sum = 0.0f;
                    std::vector<std::size_t> valid_indices;

                    for (int kh = 0; kh < window; ++kh) {
                        for (int kw = 0; kw < window; ++kw) {
                            const dim_t ih = oh * stride - padding + kh;

                            const dim_t iw = ow * stride - padding + kw;

                            if (ih < 0 || ih >= shape[2] || iw < 0 || iw >= shape[3]) {
                                continue;
                            }

                            const auto idx = offset(shape, n, c, ih, iw);

                            sum += input[idx];
                            valid_indices.push_back(idx);
                        }
                    }

                    const float denominator = count_include_pad
                                                  ? static_cast<float>(window * window)
                                                  : static_cast<float>(valid_indices.size());

                    const auto out_idx = offset(output_shape, n, c, oh, ow);

                    result.output[out_idx] = sum / denominator;

                    const float grad = upstream.empty() ? 1.0f : upstream[out_idx];

                    for (const auto idx : valid_indices) {
                        result.input_grad[idx] += grad / denominator;
                    }
                }
            }
        }
    }

    return result;
}

struct PoolConfig {
    af::dim4 shape;
    int window;
    int stride;
    int padding;
    bool count_include_pad;
};

void checkForward(const PoolConfig& config) {
    auto x = makeTensor(config.shape);

    const auto expected = referenceAvgPool(config.shape, toHost(x.data()), config.window,
                                           config.stride, config.padding, config.count_include_pad);

    auto y = x.avgpool2d(config.window, config.stride, config.padding, config.count_include_pad);

    EXPECT_EQ(y.shape(), expected.output_shape);
    expectValues(y.data(), expected.output);
}

void checkBackward(const PoolConfig& config) {
    auto x = makeTensor(config.shape, true);

    auto y = x.avgpool2d(config.window, config.stride, config.padding, config.count_include_pad);

    // Non-uniform upstream gradients detect incorrect
    // window ordering and gradient broadcasting.
    auto upstream = makeTensor(y.shape());

    const auto expected =
        referenceAvgPool(config.shape, toHost(x.data()), config.window, config.stride,
                         config.padding, config.count_include_pad, toHost(upstream.data()));

    (y * upstream).sum().backward();

    EXPECT_EQ(x.grad().dims(), config.shape);
    expectValues(x.grad(), expected.input_grad);
}

void checkNumericalGradient(const PoolConfig& config) {
    constexpr float epsilon = 1e-3f;
    constexpr float tolerance = 2e-3f;

    auto x = makeTensor(config.shape, true);

    auto y = x.avgpool2d(config.window, config.stride, config.padding, config.count_include_pad);

    auto upstream = makeTensor(y.shape());

    (y * upstream).sum().backward();

    const auto analytical = toHost(x.grad());
    auto values = toHost(x.data());

    auto evaluate = [&](const std::vector<float>& data) {
        Tensor input(af::array(config.shape, data.data()));

        auto output =
            input.avgpool2d(config.window, config.stride, config.padding, config.count_include_pad);

        return (output * upstream).sum().data().scalar<float>();
    };

    for (std::size_t i = 0; i < values.size(); ++i) {
        const float original = values[i];

        values[i] = original + epsilon;
        const float positive = evaluate(values);

        values[i] = original - epsilon;
        const float negative = evaluate(values);

        values[i] = original;

        const float numerical = (positive - negative) / (2.0f * epsilon);

        EXPECT_NEAR(analytical[i], numerical, tolerance) << "Gradient mismatch at index " << i;
    }
}

} // namespace

// ============================================================
// 1. Forward correctness
// ============================================================

TEST(AvgPool2DTest, BasicNonOverlappingPooling) {
    auto x = image(4, 4, {2, 4, 6, 8, 6, 8, 10, 12, 1, 3, 5, 7, 3, 5, 7, 9});

    auto y = x.avgpool2d(2, 2, 0, true);

    EXPECT_EQ(y.shape(), af::dim4(1, 1, 2, 2));

    // Column-major storage.
    expectValues(y.data(), {5, 3, 9, 7});
}

TEST(AvgPool2DTest, OverlappingWindows) {
    auto x = image(3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9});

    auto y = x.avgpool2d(2, 1, 0, true);

    expectValues(y.data(), {3, 6, 4, 7});
}

TEST(AvgPool2DTest, NegativeValues) {
    auto x = image(2, 2, {-1, -3, -5, -7});

    auto y = x.avgpool2d(2, 2, 0, true);

    expectValues(y.data(), {-4});
}

TEST(AvgPool2DTest, MixedSignValues) {
    auto x = image(2, 2, {-4, 2, 4, -2});

    auto y = x.avgpool2d(2, 2, 0, true);

    expectValues(y.data(), {0});
}

TEST(AvgPool2DTest, WindowSizeOneIsIdentity) {
    auto x = makeTensor(af::dim4(2, 3, 4, 5));

    auto y = x.avgpool2d(1, 1, 0, true);

    EXPECT_EQ(y.shape(), x.shape());
    expectValues(y.data(), toHost(x.data()));
}

TEST(AvgPool2DTest, WindowSizeOneWithStride) {
    checkForward({af::dim4(2, 3, 5, 7), 1, 2, 0, true});
}

TEST(AvgPool2DTest, WindowCoversEntireImage) {
    auto x = image(3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9});

    auto y = x.avgpool2d(3, 1, 0, true);

    EXPECT_EQ(y.shape(), af::dim4(1, 1, 1, 1));
    expectValues(y.data(), {5});
}

TEST(AvgPool2DTest, RectangularInput) {
    checkForward({af::dim4(1, 1, 4, 7), 2, 2, 0, true});
}

TEST(AvgPool2DTest, MultipleChannels) {
    checkForward({af::dim4(1, 4, 5, 6), 2, 1, 0, true});
}

TEST(AvgPool2DTest, MultipleBatches) {
    checkForward({af::dim4(4, 1, 5, 6), 2, 2, 0, true});
}

TEST(AvgPool2DTest, MultipleBatchesAndChannels) {
    checkForward({af::dim4(3, 4, 5, 7), 3, 2, 1, true});
}

TEST(AvgPool2DTest, FloorOutputDimensions) {
    auto x = makeTensor(af::dim4(2, 3, 5, 7));

    auto y = x.avgpool2d(2, 2, 0, true);

    EXPECT_EQ(y.shape(), af::dim4(2, 3, 2, 3));
}

TEST(AvgPool2DTest, StrideLargerThanWindow) {
    checkForward({af::dim4(2, 3, 8, 9), 2, 3, 0, true});
}

TEST(AvgPool2DTest, ConstantInputWithoutPadding) {
    auto x = constantTensor(af::dim4(2, 3, 6, 8), 7.0f);

    auto y = x.avgpool2d(3, 1, 0, true);

    expectValues(y.data(), std::vector<float>(y.data().elements(), 7.0f));
}

TEST(AvgPool2DTest, ZeroInput) {
    auto x = constantTensor(af::dim4(2, 3, 5, 7), 0.0f);

    auto y = x.avgpool2d(3, 1, 1, false);

    expectValues(y.data(), std::vector<float>(y.data().elements(), 0.0f));
}

// ============================================================
// 2. Padding semantics
// ============================================================

TEST(AvgPool2DTest, IncludePaddingKnownValues) {
    auto x = image(2, 2, {1, 2, 3, 4});

    auto y = x.avgpool2d(2, 1, 1, true);

    EXPECT_EQ(y.shape(), af::dim4(1, 1, 3, 3));

    expectValues(y.data(), {0.25f, 1.0f, 0.75f, 0.75f, 2.5f, 1.75f, 0.5f, 1.5f, 1.0f});
}

TEST(AvgPool2DTest, ExcludePaddingKnownValues) {
    auto x = image(2, 2, {1, 2, 3, 4});

    auto y = x.avgpool2d(2, 1, 1, false);

    EXPECT_EQ(y.shape(), af::dim4(1, 1, 3, 3));

    expectValues(y.data(), {1.0f, 2.0f, 3.0f, 1.5f, 2.5f, 3.5f, 2.0f, 3.0f, 4.0f});
}

TEST(AvgPool2DTest, PaddingModesEquivalentWithoutPadding) {
    auto x = makeTensor(af::dim4(2, 3, 5, 7));

    auto included = x.avgpool2d(3, 1, 0, true);
    auto excluded = x.avgpool2d(3, 1, 0, false);

    expectValues(included.data(), toHost(excluded.data()));
}

TEST(AvgPool2DTest, DefaultIncludesPadding) {
    auto x = image(2, 2, {1, 2, 3, 4});

    auto default_result = x.avgpool2d(2, 1, 1);
    auto explicit_result = x.avgpool2d(2, 1, 1, true);

    expectValues(default_result.data(), toHost(explicit_result.data()));
}

TEST(AvgPool2DTest, ExcludePaddingPreservesConstantInput) {
    auto x = constantTensor(af::dim4(2, 3, 5, 7), 4.0f);

    auto y = x.avgpool2d(3, 1, 1, false);

    expectValues(y.data(), std::vector<float>(y.data().elements(), 4.0f));
}

TEST(AvgPool2DTest, IncludePaddingReducesBorderValues) {
    auto x = constantTensor(af::dim4(1, 1, 3, 3), 9.0f);

    auto y = x.avgpool2d(3, 1, 1, true);

    expectValues(y.data(), {4, 6, 4, 6, 9, 6, 4, 6, 4});
}

TEST(AvgPool2DTest, NegativeInputWithPadding) {
    auto x = constantTensor(af::dim4(1, 1, 2, 2), -4.0f);

    auto y = x.avgpool2d(2, 1, 1, true);

    expectValues(y.data(), {-1, -2, -1, -2, -4, -2, -1, -2, -1});
}

TEST(AvgPool2DTest, ExcludePaddingMultipleChannelsAndBatches) {
    checkForward({af::dim4(3, 4, 5, 7), 3, 2, 1, false});
}

TEST(AvgPool2DTest, MaximumAllowedPadding) {
    checkForward({af::dim4(2, 3, 5, 7), 4, 2, 2, false});
}

TEST(AvgPool2DTest, WindowLargerThanUnpaddedInput) {
    checkForward({af::dim4(1, 1, 2, 2), 3, 1, 1, false});
}

TEST(AvgPool2DTest, SinglePixelIncludePadding) {
    auto x = constantTensor(af::dim4(1, 1, 1, 1), 9.0f);

    auto y = x.avgpool2d(3, 1, 1, true);

    expectValues(y.data(), {1.0f});
}

TEST(AvgPool2DTest, SinglePixelExcludePadding) {
    auto x = constantTensor(af::dim4(1, 1, 1, 1), 9.0f);

    auto y = x.avgpool2d(3, 1, 1, false);

    expectValues(y.data(), {9.0f});
}

// ============================================================
// 3. Input validation
// ============================================================

TEST(AvgPool2DTest, RejectsEmptyInput) {
    Tensor x(af::array{});

    EXPECT_THROW(x.avgpool2d(2, 1, 0, true), std::invalid_argument);
}

TEST(AvgPool2DTest, RejectsIntegerInput) {
    Tensor x(af::constant(1, af::dim4(1, 1, 4, 4), s32));

    EXPECT_THROW(x.avgpool2d(2, 1, 0, true), std::invalid_argument);
}

TEST(AvgPool2DTest, RejectsUnsignedIntegerInput) {
    Tensor x(af::constant(1, af::dim4(1, 1, 4, 4), u8));

    EXPECT_THROW(x.avgpool2d(2, 1, 0, true), std::invalid_argument);
}

TEST(AvgPool2DTest, RejectsZeroWindow) {
    auto x = makeTensor(af::dim4(1, 1, 4, 4));

    EXPECT_THROW(x.avgpool2d(0, 1, 0, true), std::invalid_argument);
}

TEST(AvgPool2DTest, RejectsNegativeWindow) {
    auto x = makeTensor(af::dim4(1, 1, 4, 4));

    EXPECT_THROW(x.avgpool2d(-2, 1, 0, true), std::invalid_argument);
}

TEST(AvgPool2DTest, RejectsZeroStride) {
    auto x = makeTensor(af::dim4(1, 1, 4, 4));

    EXPECT_THROW(x.avgpool2d(2, 0, 0, true), std::invalid_argument);
}

TEST(AvgPool2DTest, RejectsNegativeStride) {
    auto x = makeTensor(af::dim4(1, 1, 4, 4));

    EXPECT_THROW(x.avgpool2d(2, -1, 0, true), std::invalid_argument);
}

TEST(AvgPool2DTest, RejectsNegativePadding) {
    auto x = makeTensor(af::dim4(1, 1, 4, 4));

    EXPECT_THROW(x.avgpool2d(2, 1, -1, true), std::invalid_argument);
}

TEST(AvgPool2DTest, RejectsExcessivePadding) {
    auto x = makeTensor(af::dim4(1, 1, 4, 4));

    EXPECT_THROW(x.avgpool2d(3, 1, 2, false), std::invalid_argument);
}

TEST(AvgPool2DTest, RejectsWindowLargerThanHeight) {
    auto x = makeTensor(af::dim4(1, 1, 2, 6));

    EXPECT_THROW(x.avgpool2d(3, 1, 0, true), std::invalid_argument);
}

TEST(AvgPool2DTest, RejectsWindowLargerThanWidth) {
    auto x = makeTensor(af::dim4(1, 1, 6, 2));

    EXPECT_THROW(x.avgpool2d(3, 1, 0, true), std::invalid_argument);
}

TEST(AvgPool2DTest, RejectsWindowLargerThanPaddedInput) {
    auto x = makeTensor(af::dim4(1, 1, 2, 2));

    EXPECT_THROW(x.avgpool2d(5, 1, 1, false), std::invalid_argument);
}

// ============================================================
// 4. Analytical backward correctness
// ============================================================

TEST(AvgPool2DTest, BackwardNonOverlappingWindows) {
    auto x = makeTensor(af::dim4(1, 1, 4, 4), true);

    x.avgpool2d(2, 2, 0, true).sum().backward();

    expectValues(x.grad(), std::vector<float>(16, 0.25f));
}

TEST(AvgPool2DTest, BackwardOverlappingWindows) {
    auto x = makeTensor(af::dim4(1, 1, 3, 3), true);

    x.avgpool2d(2, 1, 0, true).sum().backward();

    expectValues(x.grad(), {0.25f, 0.5f, 0.25f, 0.5f, 1.0f, 0.5f, 0.25f, 0.5f, 0.25f});
}

TEST(AvgPool2DTest, BackwardIncludePadding) {
    auto x = makeTensor(af::dim4(1, 1, 2, 2), true);

    x.avgpool2d(2, 1, 1, true).sum().backward();

    expectValues(x.grad(), {1.0f, 1.0f, 1.0f, 1.0f});
}

TEST(AvgPool2DTest, BackwardExcludePadding) {
    auto x = makeTensor(af::dim4(1, 1, 2, 2), true);

    x.avgpool2d(2, 1, 1, false).sum().backward();

    expectValues(x.grad(), {2.25f, 2.25f, 2.25f, 2.25f});
}

TEST(AvgPool2DTest, BackwardSinglePixelIncludePadding) {
    auto x = constantTensor(af::dim4(1, 1, 1, 1), 9.0f, true);

    x.avgpool2d(3, 1, 1, true).sum().backward();

    expectValues(x.grad(), {1.0f / 9.0f});
}

TEST(AvgPool2DTest, BackwardSinglePixelExcludePadding) {
    auto x = constantTensor(af::dim4(1, 1, 1, 1), 9.0f, true);

    x.avgpool2d(3, 1, 1, false).sum().backward();

    expectValues(x.grad(), {1.0f});
}

TEST(AvgPool2DTest, BackwardWindowSizeOne) {
    auto x = makeTensor(af::dim4(2, 3, 4, 5), true);

    x.avgpool2d(1, 1, 0, true).sum().backward();

    expectValues(x.grad(), std::vector<float>(x.data().elements(), 1.0f));
}

TEST(AvgPool2DTest, BackwardGlobalWindow) {
    auto x = makeTensor(af::dim4(1, 1, 3, 3), true);

    x.avgpool2d(3, 1, 0, true).sum().backward();

    expectValues(x.grad(), std::vector<float>(9, 1.0f / 9.0f));
}

TEST(AvgPool2DTest, BackwardWeightedUpstream) {
    auto x = makeTensor(af::dim4(1, 1, 4, 4), true);

    auto y = x.avgpool2d(2, 2, 0, true);

    auto upstream = image(2, 2, {4, 8, 12, 16});

    (y * upstream).sum().backward();

    expectValues(x.grad(), {1, 1, 3, 3, 1, 1, 3, 3, 2, 2, 4, 4, 2, 2, 4, 4});
}

TEST(AvgPool2DTest, BackwardUnusedInputElements) {
    auto x = makeTensor(af::dim4(1, 1, 3, 3), true);

    x.avgpool2d(2, 2, 0, true).sum().backward();

    expectValues(x.grad(), {0.25f, 0.25f, 0, 0.25f, 0.25f, 0, 0, 0, 0});
}

TEST(AvgPool2DTest, BackwardStrideLargerThanWindow) {
    auto x = makeTensor(af::dim4(1, 1, 5, 5), true);

    x.avgpool2d(2, 3, 0, true).sum().backward();

    expectValues(x.grad(),
                 {0.25f, 0.25f, 0, 0.25f, 0.25f, 0.25f, 0.25f, 0, 0.25f, 0.25f, 0, 0, 0, 0, 0,
                  0.25f, 0.25f, 0, 0.25f, 0.25f, 0.25f, 0.25f, 0, 0.25f, 0.25f});
}

TEST(AvgPool2DTest, BackwardIndependentOfInputValues) {
    const af::dim4 shape(2, 3, 5, 5);

    auto x1 = constantTensor(shape, 1.0f, true);
    auto x2 = makeTensor(shape, true);

    x1.avgpool2d(3, 1, 1, false).sum().backward();
    x2.avgpool2d(3, 1, 1, false).sum().backward();

    expectValues(x1.grad(), toHost(x2.grad()));
}

// ============================================================
// 5. CPU reference comparisons
// ============================================================

class AvgPool2DReferenceTest : public testing::TestWithParam<PoolConfig> {};

TEST_P(AvgPool2DReferenceTest, Forward) {
    checkForward(GetParam());
}

TEST_P(AvgPool2DReferenceTest, Backward) {
    checkBackward(GetParam());
}

INSTANTIATE_TEST_SUITE_P(Configurations, AvgPool2DReferenceTest,
                         testing::Values(PoolConfig{af::dim4(1, 1, 4, 4), 2, 2, 0, true},
                                         PoolConfig{af::dim4(1, 1, 5, 7), 3, 1, 0, false},
                                         PoolConfig{af::dim4(2, 3, 5, 7), 2, 2, 0, true},
                                         PoolConfig{af::dim4(3, 2, 6, 8), 3, 2, 1, true},
                                         PoolConfig{af::dim4(3, 2, 6, 8), 3, 2, 1, false},
                                         PoolConfig{af::dim4(2, 4, 5, 7), 2, 1, 1, true},
                                         PoolConfig{af::dim4(2, 4, 5, 7), 2, 1, 1, false},
                                         PoolConfig{af::dim4(2, 3, 7, 9), 4, 3, 2, true},
                                         PoolConfig{af::dim4(2, 3, 7, 9), 4, 3, 2, false},
                                         PoolConfig{af::dim4(2, 3, 5, 7), 1, 2, 0, true},
                                         PoolConfig{af::dim4(1, 1, 1, 1), 3, 1, 1, false},
                                         PoolConfig{af::dim4(1, 1, 1, 1), 3, 1, 1, true},
                                         PoolConfig{af::dim4(2, 3, 1, 7), 3, 1, 1, false},
                                         PoolConfig{af::dim4(2, 3, 7, 1), 3, 1, 1, true},
                                         PoolConfig{af::dim4(4, 8, 28, 28), 2, 2, 0, true},
                                         PoolConfig{af::dim4(4, 16, 32, 32), 3, 2, 1, false}));

// ============================================================
// 6. Numerical gradient checks
// ============================================================

class AvgPool2DNumericalTest : public testing::TestWithParam<PoolConfig> {};

TEST_P(AvgPool2DNumericalTest, GradientMatchesFiniteDifferences) {
    checkNumericalGradient(GetParam());
}

INSTANTIATE_TEST_SUITE_P(Configurations, AvgPool2DNumericalTest,
                         testing::Values(PoolConfig{af::dim4(1, 1, 4, 4), 2, 2, 0, true},
                                         PoolConfig{af::dim4(1, 1, 3, 3), 2, 1, 0, true},
                                         PoolConfig{af::dim4(1, 1, 3, 4), 2, 1, 1, true},
                                         PoolConfig{af::dim4(1, 1, 3, 4), 2, 1, 1, false},
                                         PoolConfig{af::dim4(1, 2, 3, 4), 3, 2, 1, false},
                                         PoolConfig{af::dim4(2, 2, 3, 3), 2, 2, 1, true},
                                         PoolConfig{af::dim4(2, 2, 3, 3), 2, 2, 1, false},
                                         PoolConfig{af::dim4(1, 1, 3, 4), 1, 2, 0, true},
                                         PoolConfig{af::dim4(1, 1, 2, 2), 3, 1, 1, false},
                                         PoolConfig{af::dim4(1, 1, 5, 5), 2, 3, 0, true}));

// ============================================================
// 7. Autograd integration
// ============================================================

TEST(AvgPool2DTest, OutputRequiresGrad) {
    auto x = makeTensor(af::dim4(1, 1, 4, 4), true);

    EXPECT_TRUE(x.avgpool2d(2, 2, 0, true).requires_grad());
}

TEST(AvgPool2DTest, OutputDoesNotRequireGrad) {
    auto x = makeTensor(af::dim4(1, 1, 4, 4), false);

    EXPECT_FALSE(x.avgpool2d(2, 2, 0, true).requires_grad());
}

TEST(AvgPool2DTest, NoGradGuardDisablesGraph) {
    auto x = makeTensor(af::dim4(1, 1, 4, 4), true);

    neuro::NoGradGuard guard;

    EXPECT_FALSE(x.avgpool2d(2, 2, 0, true).requires_grad());
}

TEST(AvgPool2DTest, BackwardThroughInputMultiplication) {
    auto x = makeTensor(af::dim4(1, 1, 4, 4), true);

    auto y = (x * 3.0f).avgpool2d(2, 2, 0, true);

    y.sum().backward();

    expectValues(x.grad(), std::vector<float>(16, 0.75f));
}

TEST(AvgPool2DTest, BackwardThroughOutputMultiplication) {
    auto x = makeTensor(af::dim4(1, 1, 4, 4), true);

    auto y = x.avgpool2d(2, 2, 0, true) * 8.0f;

    y.sum().backward();

    expectValues(x.grad(), std::vector<float>(16, 2.0f));
}

TEST(AvgPool2DTest, BackwardThroughChainedPooling) {
    auto x = makeTensor(af::dim4(1, 1, 4, 4), true);

    auto y = x.avgpool2d(2, 2, 0, true).avgpool2d(2, 2, 0, true);

    y.sum().backward();

    expectValues(x.grad(), std::vector<float>(16, 1.0f / 16.0f));
}

TEST(AvgPool2DTest, BackwardThroughBranchingGraph) {
    auto x = makeTensor(af::dim4(1, 1, 4, 4), true);

    auto a = x.avgpool2d(2, 2, 0, true);
    auto b = x.avgpool2d(2, 2, 0, false);

    (a + b).sum().backward();

    expectValues(x.grad(), std::vector<float>(16, 0.5f));
}

TEST(AvgPool2DTest, BackwardThroughFlatten) {
    auto x = makeTensor(af::dim4(2, 3, 4, 4), true);

    auto y = x.avgpool2d(2, 2, 0, true).flatten();

    EXPECT_EQ(y.shape(), af::dim4(2, 12));

    y.sum().backward();

    expectValues(x.grad(), std::vector<float>(x.data().elements(), 0.25f));
}

TEST(AvgPool2DTest, BackwardThroughConvolution) {
    auto x = image(3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9}, true);

    auto weights = constantTensor(af::dim4(1, 1, 2, 2), 1.0f, true);

    auto y = x.conv2d(weights, 1, 0, 1).avgpool2d(2, 2, 0, true);

    expectValues(y.data(), {20.0f});

    y.sum().backward();

    expectValues(x.grad(), {0.25f, 0.5f, 0.25f, 0.5f, 1.0f, 0.5f, 0.25f, 0.5f, 0.25f});

    expectValues(weights.grad(), {3.0f, 6.0f, 4.0f, 7.0f});
}

TEST(AvgPool2DTest, ZeroGradClearsAccumulatedGradient) {
    auto x = makeTensor(af::dim4(1, 1, 4, 4), true);

    x.avgpool2d(2, 2, 0, true).sum().backward();

    x.zeroGrad();

    expectValues(x.grad(), std::vector<float>(16, 0.0f));
}

TEST(AvgPool2DTest, AccumulatesAcrossIndependentGraphs) {
    auto x = makeTensor(af::dim4(1, 1, 4, 4), true);

    x.avgpool2d(2, 2, 0, true).sum().backward();
    x.avgpool2d(2, 2, 0, true).sum().backward();

    expectValues(x.grad(), std::vector<float>(16, 0.5f));
}

TEST(AvgPool2DTest, DoesNotModifyInput) {
    auto x = makeTensor(af::dim4(2, 3, 5, 7), true);

    const auto original = toHost(x.data());

    x.avgpool2d(3, 1, 1, false).sum().backward();

    expectValues(x.data(), original);
}

// ============================================================
// 8. Floating-point precision
// ============================================================

TEST(AvgPool2DTest, PreservesFloat32) {
    auto x = makeTensor(af::dim4(1, 1, 4, 4), true);

    auto y = x.avgpool2d(2, 1, 1, false);

    EXPECT_EQ(y.data().type(), f32);

    y.sum().backward();

    EXPECT_EQ(x.grad().type(), f32);
}

TEST(AvgPool2DTest, PreservesFloat64IncludePadding) {
    Tensor x(af::constant(4.0, af::dim4(1, 1, 2, 2), f64), true);

    auto y = x.avgpool2d(2, 1, 1, true);

    EXPECT_EQ(y.data().type(), f64);

    y.sum().backward();

    EXPECT_EQ(x.grad().type(), f64);

    expectValues(x.grad(), {1, 1, 1, 1});
}

TEST(AvgPool2DTest, PreservesFloat64ExcludePadding) {
    Tensor x(af::constant(4.0, af::dim4(1, 1, 2, 2), f64), true);

    auto y = x.avgpool2d(2, 1, 1, false);

    EXPECT_EQ(y.data().type(), f64);

    y.sum().backward();

    EXPECT_EQ(x.grad().type(), f64);

    expectValues(x.grad(), {2.25f, 2.25f, 2.25f, 2.25f});
}

TEST(AvgPool2DTest, Float64NumericalPrecision) {
    const af::dim4 shape(1, 1, 2, 2);

    const double values[] = {1.000000001, 1.000000002, 1.000000003, 1.000000004};

    Tensor x(af::array(shape, values), true);

    auto y = x.avgpool2d(2, 2, 0, true);

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