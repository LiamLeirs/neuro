#include "neuro/GradMode.hpp"
#include "neuro/Tensor.hpp"
#include "neuro/nn/BatchNorm1D.hpp"

#include <arrayfire.h>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace neuro;
using namespace neuro::nn;

namespace {

std::vector<float> toHost(const af::array& array) {
    std::vector<float> values(array.elements());
    array.host(values.data());
    return values;
}

void expectValues(const af::array& array, const std::vector<float>& expected,
                  double tolerance = 1e-5) {
    const auto actual = toHost(array);

    ASSERT_EQ(actual.size(), expected.size());

    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], tolerance) << "Element " << i;
    }
}

} // namespace

TEST(BatchNorm1DTest, RejectsInvalidFeatureCount) {
    EXPECT_THROW(BatchNorm1D(0), std::invalid_argument);
    EXPECT_THROW(BatchNorm1D(-1), std::invalid_argument);
}

TEST(BatchNorm1DTest, RejectsInvalidEpsilon) {
    for (double eps : {0.0, -1e-5, std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()}) {
        EXPECT_THROW(BatchNorm1D(2, eps), std::invalid_argument);
    }
}

TEST(BatchNorm1DTest, RejectsInvalidMomentum) {
    for (double momentum : {-0.1, 1.1, std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::quiet_NaN()}) {
        EXPECT_THROW(BatchNorm1D(2, 1e-5, momentum), std::invalid_argument);
    }
}

TEST(BatchNorm1DTest, InitializesParametersAndRunningStatistics) {
    BatchNorm1D norm(3);

    EXPECT_EQ(norm.gamma().shape(), af::dim4(1, 3));
    EXPECT_EQ(norm.beta().shape(), af::dim4(1, 3));

    EXPECT_TRUE(norm.gamma().requires_grad());
    EXPECT_TRUE(norm.beta().requires_grad());

    EXPECT_EQ(norm.gamma().data().type(), f32);
    EXPECT_EQ(norm.beta().data().type(), f32);

    expectValues(norm.gamma().data(), {1, 1, 1});
    expectValues(norm.beta().data(), {0, 0, 0});

    EXPECT_EQ(norm.runningMean().dims(), af::dim4(1, 3));
    EXPECT_EQ(norm.runningVar().dims(), af::dim4(1, 3));

    expectValues(norm.runningMean(), {0, 0, 0});
    expectValues(norm.runningVar(), {1, 1, 1});

    EXPECT_TRUE(norm.isTraining());
}

TEST(BatchNorm1DTest, ExposesOnlyTrainableParameters) {
    BatchNorm1D norm(3);

    const auto parameters = norm.parameters();

    ASSERT_EQ(parameters.size(), 2u);

    EXPECT_TRUE(parameters[0].requires_grad());
    EXPECT_TRUE(parameters[1].requires_grad());

    expectValues(parameters[0].data(), {1, 1, 1});
    expectValues(parameters[1].data(), {0, 0, 0});
}

TEST(BatchNorm1DTest, RejectsInvalidInput) {
    BatchNorm1D norm(3);

    Tensor empty;
    Tensor wrong_features(af::constant(1, 4, 2));
    Tensor wrong_rank(af::constant(1, 4, 3, 2, 2));
    Tensor integer_input(af::constant(1, af::dim4(4, 3), s32));
    Tensor double_input(af::constant(1, af::dim4(4, 3), f64));

    EXPECT_THROW(norm(empty), std::invalid_argument);
    EXPECT_THROW(norm(wrong_features), std::invalid_argument);
    EXPECT_THROW(norm(wrong_rank), std::invalid_argument);
    EXPECT_THROW(norm(integer_input), std::invalid_argument);
    EXPECT_THROW(norm(double_input), std::invalid_argument);
}

TEST(BatchNorm1DTest, RejectsSingletonDuringTraining) {
    BatchNorm1D norm(3);

    Tensor input(af::constant(1, 1, 3));

    EXPECT_THROW(norm(input), std::invalid_argument);
}

TEST(BatchNorm1DTest, NormalizesFeaturesIndependently) {
    BatchNorm1D norm(2);

    // Feature 0: [1, 2, 3]
    // Feature 1: [10, 20, 30]
    const float values[]{1, 2, 3, 10, 20, 30};

    Tensor input(af::array(3, 2, values));

    Tensor output = norm(input);

    EXPECT_EQ(output.shape(), input.shape());
    EXPECT_EQ(output.data().type(), f32);

    const float a = 1.0 / std::sqrt(2.0 / 3.0 + 1e-5);
    const float b = 10.0 / std::sqrt(200.0 / 3.0 + 1e-5);

    expectValues(output.data(), {-a, 0, a, -b, 0, b});
}

TEST(BatchNorm1DTest, HandlesConstantFeatures) {
    BatchNorm1D norm(2);

    const float values[]{3, 3, 3, 7, 7, 7};

    Tensor input(af::array(3, 2, values), true);

    Tensor output = norm(input);

    expectValues(output.data(), {0, 0, 0, 0, 0, 0}, 2e-4);

    output.sum().backward();

    expectValues(input.grad(), {0, 0, 0, 0, 0, 0});
}

TEST(BatchNorm1DTest, AppliesAffineTransformation) {
    BatchNorm1D norm(2);

    const float gamma[]{2, 3};
    const float beta[]{1, -2};

    norm.gamma() = Tensor(af::array(1, 2, gamma), true);
    norm.beta() = Tensor(af::array(1, 2, beta), true);

    const float values[]{1, 3, 10, 14};

    Tensor input(af::array(2, 2, values));

    const float a = 1.0 / std::sqrt(1.0 + 1e-5);
    const float b = 2.0 / std::sqrt(4.0 + 1e-5);

    expectValues(norm(input).data(), {1 - 2 * a, 1 + 2 * a, -2 - 3 * b, -2 + 3 * b});
}

TEST(BatchNorm1DTest, UpdatesRunningStatistics) {
    BatchNorm1D norm(2, 1e-5, 0.1);

    // Means: [2, 20]
    // Unbiased variances: [1, 100]
    const float values[]{1, 2, 3, 10, 20, 30};

    Tensor input(af::array(3, 2, values));

    norm(input);

    expectValues(norm.runningMean(), {0.2f, 2.0f});
    expectValues(norm.runningVar(), {1.0f, 10.9f});

    norm(input);

    expectValues(norm.runningMean(), {0.38f, 3.8f});
    expectValues(norm.runningVar(), {1.0f, 19.81f});
}

TEST(BatchNorm1DTest, ZeroMomentumPreservesRunningStatistics) {
    BatchNorm1D norm(2, 1e-5, 0.0);

    const float values[]{1, 2, 3, 10, 20, 30};

    norm(Tensor(af::array(3, 2, values)));

    expectValues(norm.runningMean(), {0, 0});
    expectValues(norm.runningVar(), {1, 1});
}

TEST(BatchNorm1DTest, FullMomentumReplacesRunningStatistics) {
    BatchNorm1D norm(2, 1e-5, 1.0);

    const float values[]{1, 2, 3, 10, 20, 30};

    norm(Tensor(af::array(3, 2, values)));

    expectValues(norm.runningMean(), {2, 20});
    expectValues(norm.runningVar(), {1, 100});
}

TEST(BatchNorm1DTest, EvaluationUsesRunningStatistics) {
    BatchNorm1D norm(2, 1e-5, 1.0);

    const float training_values[]{1, 2, 3, 10, 20, 30};

    norm(Tensor(af::array(3, 2, training_values)));

    norm.eval();

    // Single-sample inference.
    const float values[]{4, 40};

    Tensor input(af::array(1, 2, values));

    Tensor output = norm(input);

    EXPECT_EQ(output.shape(), af::dim4(1, 2));

    expectValues(output.data(), {static_cast<float>(2.0 / std::sqrt(1.0 + 1e-5)),
                                 static_cast<float>(20.0 / std::sqrt(100.0 + 1e-5))});

    expectValues(norm.runningMean(), {2, 20});
    expectValues(norm.runningVar(), {1, 100});
}

TEST(BatchNorm1DTest, EvaluationBeforeTrainingUsesInitialStatistics) {
    BatchNorm1D norm(2);
    norm.eval();

    const float values[]{2, 4};

    Tensor input(af::array(1, 2, values));

    const float scale = 1.0 / std::sqrt(1.0 + 1e-5);

    expectValues(norm(input).data(), {2 * scale, 4 * scale});
}

TEST(BatchNorm1DTest, EvaluationIsIndependentOfBatchComposition) {
    BatchNorm1D norm(2);
    norm.eval();

    const float single_values[]{2, 4};
    const float batch_values[]{2, 100, 4, -100};

    Tensor single(af::array(1, 2, single_values));
    Tensor batch(af::array(2, 2, batch_values));

    const auto single_output = toHost(norm(single).data());
    const auto batch_output = toHost(norm(batch).data());

    ASSERT_EQ(single_output.size(), 2u);
    ASSERT_EQ(batch_output.size(), 4u);

    EXPECT_NEAR(single_output[0], batch_output[0], 1e-6);
    EXPECT_NEAR(single_output[1], batch_output[2], 1e-6);
}

TEST(BatchNorm1DTest, ComputesAffineParameterGradients) {
    BatchNorm1D norm(2);

    const float values[]{1, 3, 10, 14};
    const float upstream_values[]{1, 3, 2, 6};

    Tensor input(af::array(2, 2, values), true);
    Tensor upstream(af::array(2, 2, upstream_values));

    (norm(input) * upstream).sum().backward();

    const float a = 1.0 / std::sqrt(1.0 + 1e-5);
    const float b = 2.0 / std::sqrt(4.0 + 1e-5);

    expectValues(norm.gamma().grad(), {2 * a, 4 * b});
    expectValues(norm.beta().grad(), {4, 8});
}

TEST(BatchNorm1DTest, ComputesInputGradients) {
    constexpr double eps = 1e-5;

    BatchNorm1D norm(1, eps);

    const float values[]{1, 2, 4};
    const float upstream_values[]{1, -2, 3};

    Tensor input(af::array(3, 1, values), true);
    Tensor upstream(af::array(3, 1, upstream_values));

    (norm(input) * upstream).sum().backward();

    constexpr double n = 3.0;
    const double mean = 7.0 / 3.0;
    const double variance = 14.0 / 9.0;
    const double inv_std = 1.0 / std::sqrt(variance + eps);

    double sum_grad = 0;
    double sum_grad_normalized = 0;

    for (int i = 0; i < 3; ++i) {
        const double normalized = (values[i] - mean) * inv_std;

        sum_grad += upstream_values[i];
        sum_grad_normalized += upstream_values[i] * normalized;
    }

    std::vector<float> expected;

    for (int i = 0; i < 3; ++i) {
        const double normalized = (values[i] - mean) * inv_std;

        const double gradient =
            inv_std / n * (n * upstream_values[i] - sum_grad - normalized * sum_grad_normalized);

        expected.push_back(static_cast<float>(gradient));
    }

    expectValues(input.grad(), expected);
}

TEST(BatchNorm1DTest, EvaluationSupportsBackward) {
    BatchNorm1D norm(2);
    norm.eval();

    const float values[]{2, 4};

    Tensor input(af::array(1, 2, values), true);

    norm(input).sum().backward();

    const float scale = 1.0 / std::sqrt(1.0 + 1e-5);

    expectValues(input.grad(), {scale, scale});
    expectValues(norm.gamma().grad(), {2 * scale, 4 * scale});
    expectValues(norm.beta().grad(), {1, 1});
}

TEST(BatchNorm1DTest, RespectsNoGradGuard) {
    BatchNorm1D norm(1, 1e-5, 1.0);

    Tensor input({1.0f, 2.0f, 3.0f}, true);

    NoGradGuard guard;

    Tensor output = norm(input);

    EXPECT_FALSE(output.requires_grad());

    // Gradient tracking is independent of training mode.
    expectValues(norm.runningMean(), {2});
    expectValues(norm.runningVar(), {1});
}

TEST(BatchNorm1DTest, NormalizesThreeDimensionalInput) {
    BatchNorm1D norm(2, 1e-5, 1.0);

    // Shape: (N=2, C=2, L=2)
    // Channel 0: [1, 3, 5, 7]
    // Channel 1: [10, 30, 50, 70]
    const float values[]{1, 3, 10, 30, 5, 7, 50, 70};

    Tensor input(af::array(2, 2, 2, values));
    Tensor output = norm(input);

    EXPECT_EQ(output.shape(), input.shape());

    const float a = 1.0 / std::sqrt(5.0 + 1e-5);
    const float b = 1.0 / std::sqrt(500.0 + 1e-5);

    expectValues(output.data(), {-3 * a, -a, -30 * b, -10 * b, a, 3 * a, 10 * b, 30 * b});

    expectValues(norm.runningMean(), {4, 40});
    expectValues(norm.runningVar(), {20.0f / 3.0f, 2000.0f / 3.0f}, 1e-4);
}

TEST(BatchNorm1DTest, SupportsSingletonBatchWithSequence) {
    BatchNorm1D norm(1);

    const float values[]{1, 3, 5};

    Tensor input(af::array(1, 1, 3, values));

    const float scale = 2.0 / std::sqrt(8.0 / 3.0 + 1e-5);

    expectValues(norm(input).data(), {-scale, 0, scale});
}

TEST(BatchNorm1DTest, EvaluatesThreeDimensionalInput) {
    BatchNorm1D norm(1, 1e-5, 1.0);

    const float training_values[]{1, 3, 5, 7};

    norm(Tensor(af::array(2, 1, 2, training_values)));

    norm.eval();

    const float values[]{2, 4, 6};
    Tensor input(af::array(1, 1, 3, values));

    const float scale = 1.0 / std::sqrt(20.0 / 3.0 + 1e-5);

    expectValues(norm(input).data(), {-2 * scale, 0, 2 * scale});

    expectValues(norm.runningMean(), {4});
    expectValues(norm.runningVar(), {20.0f / 3.0f});
}

TEST(BatchNorm1DTest, ComputesThreeDimensionalGradients) {
    BatchNorm1D norm(1);

    const float values[]{1, 3, 5, 7};
    const float weights[]{1, -2, 3, 4};

    Tensor input(af::array(2, 1, 2, values), true);
    Tensor upstream(af::array(2, 1, 2, weights));

    (norm(input) * upstream).sum().backward();

    const double inv_std = 1.0 / std::sqrt(5.0 + 1e-5);

    std::vector<float> expected;

    for (int i = 0; i < 4; ++i) {
        const double centered = values[i] - 4.0;

        expected.push_back(static_cast<float>(
            inv_std * (weights[i] - 1.5 - centered * 14.0 / (4.0 * (5.0 + 1e-5)))));
    }

    expectValues(input.grad(), expected);
    expectValues(norm.gamma().grad(), {static_cast<float>(14.0 * inv_std)});
    expectValues(norm.beta().grad(), {6});
}