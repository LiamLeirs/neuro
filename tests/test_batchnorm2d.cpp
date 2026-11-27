#include "neuro/Tensor.hpp"
#include "neuro/nn/BatchNorm2D.hpp"

#include <af/array.h>
#include <af/data.h>
#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

using namespace neuro;
using namespace neuro::nn;

namespace {

void expectValues(const af::array& actual, const std::vector<float>& expected,
                  double tolerance = 1e-5) {
    ASSERT_EQ(actual.elements(), expected.size());

    std::vector<float> values(actual.elements());
    actual.host(values.data());

    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(values[i], expected[i], tolerance) << "Index: " << i;
    }
}

} // namespace

TEST(BatchNorm2DTest, RejectsInvalidInput) {
    BatchNorm2D norm(3);

    EXPECT_THROW(norm(Tensor()), std::invalid_argument);

    Tensor wrong_channels(af::constant(1, 2, 4, 3, 3));
    EXPECT_THROW(norm(wrong_channels), std::invalid_argument);

    Tensor wrong_dtype(af::constant(1, 2, 3, 3, 3, s32));
    EXPECT_THROW(norm(wrong_dtype), std::invalid_argument);
}

TEST(BatchNorm2DTest, RejectsSingleValuePerChannel) {
    BatchNorm2D norm(3);

    Tensor input(af::constant(1, 1, 3, 1, 1));

    EXPECT_THROW(norm(input), std::invalid_argument);
}

TEST(BatchNorm2DTest, NormalizesChannelsIndependently) {
    BatchNorm2D norm(2, 1e-5, 1.0);

    // Shape: (N=2, C=2, H=2, W=2)
    const float values[]{1, 2, 10, 20, 3, 4, 30, 40, 5, 6, 50, 60, 7, 8, 70, 80};

    Tensor input(af::array(2, 2, 2, 2, values));
    Tensor output = norm(input);

    EXPECT_EQ(output.shape(), input.shape());

    std::vector<float> expected;

    for (int i = 0; i < 16; ++i) {
        const int channel = (i / 2) % 2;
        const double mean = channel == 0 ? 4.5 : 45.0;
        const double var = channel == 0 ? 5.25 : 525.0;

        expected.push_back((values[i] - mean) / std::sqrt(var + 1e-5));
    }

    expectValues(output.data(), expected);
    expectValues(norm.runningMean(), {4.5f, 45.0f});
    expectValues(norm.runningVar(), {6.0f, 600.0f}, 1e-4);
}

TEST(BatchNorm2DTest, SupportsNonSquareSpatialDimensions) {
    BatchNorm2D norm(1);

    // Shape: (1, 1, 2, 3)
    const float values[]{1, 2, 3, 4, 5, 6};

    Tensor input(af::array(1, 1, 2, 3, values));
    Tensor output = norm(input);

    EXPECT_EQ(output.shape(), input.shape());

    std::vector<float> expected;

    for (float value : values) {
        expected.push_back((value - 3.5) / std::sqrt(35.0 / 12.0 + 1e-5));
    }

    expectValues(output.data(), expected);
}

TEST(BatchNorm2DTest, SupportsSingletonSpatialDimensions) {
    BatchNorm2D norm(1);

    const float values[]{1, 3, 5};

    Tensor input(af::array(3, 1, 1, 1, values));

    const float scale = 2.0 / std::sqrt(8.0 / 3.0 + 1e-5);

    expectValues(norm(input).data(), {-scale, 0, scale});
}

TEST(BatchNorm2DTest, AppliesAffineTransformation) {
    BatchNorm2D norm(1);

    norm.gamma() = Tensor(af::constant(2, 1, 1), true);
    norm.beta() = Tensor(af::constant(3, 1, 1), true);

    const float values[]{1, 3, 5, 7};

    Tensor input(af::array(1, 1, 2, 2, values));

    const float scale = 2.0 / std::sqrt(5.0 + 1e-5);

    expectValues(norm(input).data(), {3 - 3 * scale, 3 - scale, 3 + scale, 3 + 3 * scale});
}

TEST(BatchNorm2DTest, EvaluationUsesRunningStatistics) {
    BatchNorm2D norm(1, 1e-5, 1.0);

    const float training_values[]{1, 3, 5, 7};

    norm(Tensor(af::array(1, 1, 2, 2, training_values)));

    norm.eval();

    const float values[]{2, 4, 6};

    Tensor input(af::array(1, 1, 1, 3, values));

    const float scale = 1.0 / std::sqrt(20.0 / 3.0 + 1e-5);

    expectValues(norm(input).data(), {-2 * scale, 0, 2 * scale});

    expectValues(norm.runningMean(), {4});
    expectValues(norm.runningVar(), {20.0f / 3.0f});

    // Evaluation must also accept one value per channel.
    Tensor singleton(af::constant(4, 1, 1, 1, 1));

    expectValues(norm(singleton).data(), {0});
}

TEST(BatchNorm2DTest, ComputesInputAndParameterGradients) {
    BatchNorm2D norm(1);

    const float values[]{1, 3, 5, 7};
    const float weights[]{1, -2, 3, 4};

    Tensor input(af::array(1, 1, 2, 2, values), true);
    Tensor upstream(af::array(1, 1, 2, 2, weights));

    (norm(input) * upstream).sum().backward();

    const double inv_std = 1.0 / std::sqrt(5.0 + 1e-5);

    std::vector<float> expected;

    for (int i = 0; i < 4; ++i) {
        const double centered = values[i] - 4.0;

        expected.push_back(inv_std * (weights[i] - 1.5 - centered * 14.0 / (4.0 * (5.0 + 1e-5))));
    }

    expectValues(input.grad(), expected);

    expectValues(norm.gamma().grad(), {static_cast<float>(14.0 * inv_std)});

    expectValues(norm.beta().grad(), {6});
}