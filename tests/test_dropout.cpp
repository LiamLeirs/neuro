#include "neuro/GradMode.hpp"
#include "neuro/Tensor.hpp"
#include "neuro/nn/Dropout.hpp"

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

} // namespace

TEST(DropoutTest, RejectsInvalidProbability) {
    Tensor input({1.0f, 2.0f, 3.0f});

    for (double p :
         {-0.1, 1.1, std::numeric_limits<double>::infinity(),
          -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        EXPECT_THROW(input.dropout(p), std::invalid_argument);
        EXPECT_THROW(Dropout{p}, std::invalid_argument);
    }
}

TEST(DropoutTest, RejectsIntegerInput) {
    Tensor input(af::constant(1, af::dim4(10), s32));

    EXPECT_THROW(input.dropout(0.5), std::invalid_argument);
}

TEST(DropoutTest, RejectsEmptyInput) {
    Tensor input{af::array{}};

    EXPECT_THROW(input.dropout(0.5), std::invalid_argument);
}

TEST(DropoutTest, ZeroProbabilityPreservesInput) {
    Tensor input({1.0f, -2.0f, 3.0f}, true);

    Tensor output = input.dropout(0.0);

    EXPECT_EQ(output.shape(), input.shape());
    EXPECT_EQ(toHost(output.data()), toHost(input.data()));

    output.sum().backward();

    EXPECT_EQ(toHost(input.grad()), (std::vector<float>{1.0f, 1.0f, 1.0f}));
}

TEST(DropoutTest, FullProbabilityProducesZeroOutputAndGradient) {
    Tensor input({1.0f, -2.0f, 3.0f}, true);

    Tensor output = input.dropout(1.0);

    EXPECT_EQ(output.shape(), input.shape());
    EXPECT_TRUE(output.requires_grad());

    EXPECT_EQ(toHost(output.data()), (std::vector<float>{0.0f, 0.0f, 0.0f}));

    output.sum().backward();

    EXPECT_EQ(toHost(input.grad()), (std::vector<float>{0.0f, 0.0f, 0.0f}));
}

TEST(DropoutTest, PreservesShapeAndDtype) {
    const af::dim4 shape(2, 3, 4, 5);

    for (auto type : {f32, f64}) {
        Tensor input(af::constant(1.0, shape, type));

        Tensor output = input.dropout(0.5);

        EXPECT_EQ(output.shape(), shape);
        EXPECT_EQ(output.data().type(), type);
    }
}

TEST(DropoutTest, AppliesInvertedDropoutScaling) {
    Tensor input(af::constant(3.0f, af::dim4(1024), f32));

    Tensor output = input.dropout(0.25);

    for (float value : toHost(output.data())) {
        if (value != 0.0f) {
            EXPECT_NEAR(value, 4.0f, 1e-6);
        }
    }
}

TEST(DropoutTest, DropRateApproximatesProbability) {
    constexpr int size = 20000;
    constexpr double probability = 0.3;

    Tensor input(af::constant(1.0f, af::dim4(size), f32));

    Tensor output = input.dropout(probability);

    std::size_t dropped = 0;

    for (float value : toHost(output.data())) {
        if (value == 0.0f) {
            ++dropped;
        }
    }

    const double drop_rate = static_cast<double>(dropped) / size;

    EXPECT_NEAR(drop_rate, probability, 0.02);
}

TEST(DropoutTest, BackwardUsesForwardMask) {
    Tensor input(af::constant(1.0f, af::dim4(1024), f32), true);

    Tensor output = input.dropout(0.5);

    const auto output_values = toHost(output.data());

    output.sum().backward();

    const auto gradients = toHost(input.grad());

    ASSERT_EQ(gradients.size(), output_values.size());

    for (std::size_t i = 0; i < gradients.size(); ++i) {
        EXPECT_FLOAT_EQ(gradients[i], output_values[i]);
    }
}

TEST(DropoutTest, BackwardScalesUpstreamGradient) {
    const af::dim4 shape(1024);

    Tensor input(af::constant(1.0f, shape, f32), true);
    Tensor upstream(af::randu(shape, f32));

    Tensor output = input.dropout(0.5);

    (output * upstream).sum().backward();

    const auto output_values = toHost(output.data());
    const auto upstream_values = toHost(upstream.data());
    const auto gradients = toHost(input.grad());

    for (std::size_t i = 0; i < gradients.size(); ++i) {
        EXPECT_NEAR(gradients[i], output_values[i] * upstream_values[i], 1e-6);
    }
}

TEST(DropoutTest, RespectsNoGradGuard) {
    Tensor input(af::constant(1.0f, af::dim4(128), f32), true);

    NoGradGuard guard;

    Tensor output = input.dropout(0.5);

    EXPECT_FALSE(output.requires_grad());
}

TEST(DropoutModuleTest, HasNoTrainableParameters) {
    Dropout dropout;

    EXPECT_TRUE(dropout.parameters().empty());
}

TEST(DropoutModuleTest, EvaluationPreservesInputAndGradient) {
    Dropout dropout(0.5);
    dropout.eval();

    Tensor input({1.0f, -2.0f, 3.0f}, true);

    Tensor output = dropout(input);

    EXPECT_EQ(toHost(output.data()), toHost(input.data()));

    output.sum().backward();

    EXPECT_EQ(toHost(input.grad()), (std::vector<float>{1.0f, 1.0f, 1.0f}));
}

TEST(DropoutModuleTest, TrainingModeAppliesDropout) {
    Dropout dropout(1.0);

    Tensor input({1.0f, 2.0f, 3.0f});

    EXPECT_TRUE(dropout.isTraining());

    EXPECT_EQ(toHost(dropout(input).data()), (std::vector<float>{0.0f, 0.0f, 0.0f}));

    dropout.eval();

    EXPECT_EQ(toHost(dropout(input).data()), toHost(input.data()));

    dropout.train();

    EXPECT_EQ(toHost(dropout(input).data()), (std::vector<float>{0.0f, 0.0f, 0.0f}));
}