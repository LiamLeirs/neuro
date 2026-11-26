#include "neuro/GradMode.hpp"
#include "neuro/Tensor.hpp"

#include <arrayfire.h>
#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

using namespace neuro;

namespace {

template <typename T = float> std::vector<T> toHost(const af::array& array) {
    std::vector<T> values(array.elements());
    array.host(values.data());
    return values;
}

} // namespace

TEST(VarianceTest, ComputesPopulationVariance) {
    Tensor input({1.0f, 2.0f, 3.0f, 4.0f});

    Tensor output = input.var();

    EXPECT_EQ(output.data().elements(), 1);
    EXPECT_NEAR(output.data().scalar<float>(), 1.25f, 1e-6);
}

TEST(VarianceTest, ConstantInputHasZeroVariance) {
    Tensor input({5.0f, 5.0f, 5.0f});

    EXPECT_FLOAT_EQ(input.var().data().scalar<float>(), 0.0f);
}

TEST(VarianceTest, SingletonHasZeroVariance) {
    Tensor input({7.0f}, true);

    Tensor output = input.var();

    EXPECT_FLOAT_EQ(output.data().scalar<float>(), 0.0f);

    output.backward();

    EXPECT_FLOAT_EQ(input.grad().scalar<float>(), 0.0f);
}

TEST(VarianceTest, Backward) {
    Tensor input({1.0f, 2.0f, 3.0f, 4.0f}, true);

    input.var().backward();

    const auto gradients = toHost(input.grad());
    const std::vector<float> expected{-0.75f, -0.25f, 0.25f, 0.75f};

    ASSERT_EQ(gradients.size(), expected.size());

    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(gradients[i], expected[i], 1e-6);
    }
}

TEST(VarianceTest, ReducesAlongDimension) {
    const float values[]{1.0f, 3.0f, 2.0f, 6.0f};

    Tensor input(af::array(2, 2, values));

    Tensor output = input.var(0);

    EXPECT_EQ(output.shape(), af::dim4(1, 2));

    const auto result = toHost(output.data());

    ASSERT_EQ(result.size(), 2u);
    EXPECT_NEAR(result[0], 1.0f, 1e-6);
    EXPECT_NEAR(result[1], 4.0f, 1e-6);
}

TEST(VarianceTest, ReducesAlongSecondDimension) {
    const float values[]{1.0f, 3.0f, 2.0f, 6.0f};

    Tensor input(af::array(2, 2, values));

    Tensor output = input.var(1);

    EXPECT_EQ(output.shape(), af::dim4(2, 1));

    const auto result = toHost(output.data());

    ASSERT_EQ(result.size(), 2u);
    EXPECT_NEAR(result[0], 0.25f, 1e-6);
    EXPECT_NEAR(result[1], 2.25f, 1e-6);
}

TEST(VarianceTest, DimensionBackward) {
    const float values[]{1.0f, 3.0f, 2.0f, 6.0f};

    Tensor input(af::array(2, 2, values), true);

    input.var(0).sum().backward();

    const auto gradients = toHost(input.grad());
    const std::vector<float> expected{-1.0f, 1.0f, -2.0f, 2.0f};

    ASSERT_EQ(gradients.size(), expected.size());

    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(gradients[i], expected[i], 1e-6);
    }
}

TEST(VarianceTest, SupportsDoublePrecision) {
    const double values[]{1.0, 2.0, 3.0, 4.0};

    Tensor input(af::array(4, values), true);

    Tensor output = input.var();

    EXPECT_EQ(output.data().type(), f64);
    EXPECT_NEAR(output.data().scalar<double>(), 1.25, 1e-12);

    output.backward();

    EXPECT_EQ(input.grad().type(), f64);

    const auto gradients = toHost<double>(input.grad());

    ASSERT_EQ(gradients.size(), 4u);
    EXPECT_NEAR(gradients[0], -0.75, 1e-12);
    EXPECT_NEAR(gradients[3], 0.75, 1e-12);
}

TEST(VarianceTest, RejectsInvalidDimensions) {
    Tensor input({1.0f, 2.0f});

    EXPECT_THROW(input.var(-1), std::out_of_range);
    EXPECT_THROW(input.var(4), std::out_of_range);
}

TEST(VarianceTest, RejectsIntegerInput) {
    Tensor input(af::constant(1, af::dim4(4), s32));

    EXPECT_THROW(input.var(), std::invalid_argument);
    EXPECT_THROW(input.var(0), std::invalid_argument);
}

TEST(VarianceTest, RejectsEmptyInput) {
    Tensor input{af::array{}};

    EXPECT_THROW(input.var(), std::invalid_argument);
    EXPECT_THROW(input.var(0), std::invalid_argument);
}

TEST(SqrtTest, ComputesSquareRoot) {
    Tensor input({0.0f, 1.0f, 4.0f, 9.0f, 16.0f});

    Tensor output = input.sqrt();

    EXPECT_EQ(output.shape(), input.shape());

    EXPECT_EQ(toHost(output.data()), (std::vector<float>{0.0f, 1.0f, 2.0f, 3.0f, 4.0f}));
}

TEST(SqrtTest, Backward) {
    Tensor input({1.0f, 4.0f, 9.0f, 16.0f}, true);

    input.sqrt().sum().backward();

    const auto gradients = toHost(input.grad());
    const std::vector<float> expected{0.5f, 0.25f, 1.0f / 6.0f, 0.125f};

    ASSERT_EQ(gradients.size(), expected.size());

    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(gradients[i], expected[i], 1e-6);
    }
}

TEST(SqrtTest, BackwardScalesUpstreamGradient) {
    Tensor input({1.0f, 4.0f, 9.0f}, true);
    Tensor upstream({2.0f, 8.0f, -6.0f});

    (input.sqrt() * upstream).sum().backward();

    const auto gradients = toHost(input.grad());
    const std::vector<float> expected{1.0f, 2.0f, -1.0f};

    ASSERT_EQ(gradients.size(), expected.size());

    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(gradients[i], expected[i], 1e-6);
    }
}

TEST(SqrtTest, NegativeInputProducesNaN) {
    Tensor input({-1.0f, -4.0f});

    const auto result = toHost(input.sqrt().data());

    ASSERT_EQ(result.size(), 2u);
    EXPECT_TRUE(std::isnan(result[0]));
    EXPECT_TRUE(std::isnan(result[1]));
}

TEST(SqrtTest, ZeroHasInfiniteGradient) {
    Tensor input({0.0f}, true);

    input.sqrt().backward();

    const float gradient = input.grad().scalar<float>();

    EXPECT_TRUE(std::isinf(gradient));
    EXPECT_GT(gradient, 0.0f);
}

TEST(SqrtTest, SupportsDoublePrecision) {
    const double values[]{2.0, 4.0};

    Tensor input(af::array(2, values), true);

    Tensor output = input.sqrt();

    EXPECT_EQ(output.data().type(), f64);

    const auto result = toHost<double>(output.data());

    ASSERT_EQ(result.size(), 2u);
    EXPECT_NEAR(result[0], std::sqrt(2.0), 1e-12);
    EXPECT_NEAR(result[1], 2.0, 1e-12);

    output.sum().backward();

    EXPECT_EQ(input.grad().type(), f64);

    const auto gradients = toHost<double>(input.grad());

    ASSERT_EQ(gradients.size(), 2u);
    EXPECT_NEAR(gradients[0], 1.0 / (2.0 * std::sqrt(2.0)), 1e-12);
    EXPECT_NEAR(gradients[1], 0.25, 1e-12);
}

TEST(SqrtTest, RejectsIntegerInput) {
    Tensor input(af::constant(4, af::dim4(4), s32));

    EXPECT_THROW(input.sqrt(), std::invalid_argument);
}

TEST(SqrtTest, RejectsEmptyInput) {
    Tensor input{af::array{}};

    EXPECT_THROW(input.sqrt(), std::invalid_argument);
}

TEST(VarianceSqrtTest, RespectsNoGradGuard) {
    Tensor input({1.0f, 2.0f, 3.0f}, true);

    NoGradGuard guard;

    EXPECT_FALSE(input.var().requires_grad());
    EXPECT_FALSE(input.var(0).requires_grad());
    EXPECT_FALSE(input.sqrt().requires_grad());
}