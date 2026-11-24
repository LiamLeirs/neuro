#include "neuro/Tensor.hpp"
#include "neuro/metrics/Mean.hpp"

#include <af/algorithm.h>
#include <af/data.h>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

using namespace neuro;
using namespace neuro::metrics;

// ==================== Basic Computation ====================

TEST(MeanTest, SingleValue) {
    Mean mean;

    mean.update(5.0);

    EXPECT_DOUBLE_EQ(mean.compute(), 5.0);
}

TEST(MeanTest, MultipleValues) {
    Mean mean;

    mean.update(2.0);
    mean.update(4.0);
    mean.update(6.0);

    EXPECT_DOUBLE_EQ(mean.compute(), 4.0);
}

TEST(MeanTest, NegativeValues) {
    Mean mean;

    mean.update(-2.0);
    mean.update(-4.0);
    mean.update(-6.0);

    EXPECT_DOUBLE_EQ(mean.compute(), -4.0);
}

TEST(MeanTest, MixedPositiveAndNegativeValues) {
    Mean mean;

    mean.update(-5.0);
    mean.update(5.0);

    EXPECT_DOUBLE_EQ(mean.compute(), 0.0);
}

TEST(MeanTest, DoublePrecision) {
    Mean mean;

    const double value = 1.0000000001;

    mean.update(value);

    EXPECT_DOUBLE_EQ(mean.compute(), value);
}

// ==================== Weighted Mean ====================

TEST(MeanTest, EqualWeights) {
    Mean mean;

    mean.update(2.0, 5);
    mean.update(6.0, 5);

    EXPECT_DOUBLE_EQ(mean.compute(), 4.0);
}

TEST(MeanTest, UnequalWeights) {
    Mean mean;

    mean.update(2.0, 3);
    mean.update(6.0, 1);

    EXPECT_DOUBLE_EQ(mean.compute(), 3.0);
}

TEST(MeanTest, MixedWeightedAndUnweightedValues) {
    Mean mean;

    mean.update(2.0);
    mean.update(6.0, 3);

    EXPECT_DOUBLE_EQ(mean.compute(), 5.0);
}

TEST(MeanTest, UnequalBatchSizes) {
    Mean mean;

    mean.update(0.2, 128);
    mean.update(0.8, 32);

    EXPECT_NEAR(mean.compute(), 0.32, 1e-12);
}

// ==================== Tensor Support ====================

TEST(MeanTest, Float32Scalar) {
    Mean mean;

    Tensor value({2.5f});

    mean.update(value);

    EXPECT_DOUBLE_EQ(mean.compute(), 2.5);
}

TEST(MeanTest, Float64Scalar) {
    Mean mean;

    const double expected = 1.0000000001;
    Tensor value(af::constant(expected, 1, f64));

    mean.update(value);

    EXPECT_DOUBLE_EQ(mean.compute(), expected);
}

TEST(MeanTest, SignedIntegerScalar) {
    Mean mean;

    Tensor value(af::constant(-7, 1, s32));

    mean.update(value);

    EXPECT_DOUBLE_EQ(mean.compute(), -7.0);
}

TEST(MeanTest, UnsignedIntegerScalar) {
    Mean mean;

    Tensor value(af::constant(42, 1, u32));

    mean.update(value);

    EXPECT_DOUBLE_EQ(mean.compute(), 42.0);
}

TEST(MeanTest, WeightedTensorValues) {
    Mean mean;

    Tensor first({2.0f});
    Tensor second({6.0f});

    mean.update(first, 3);
    mean.update(second, 1);

    EXPECT_DOUBLE_EQ(mean.compute(), 3.0);
}

TEST(MeanTest, MixedTensorAndDoubleValues) {
    Mean mean;

    Tensor value({2.0f});

    mean.update(value, 3);
    mean.update(6.0);

    EXPECT_DOUBLE_EQ(mean.compute(), 3.0);
}

// ==================== Edge Cases ====================

TEST(MeanTest, EmptyStateReturnsZero) {
    Mean mean;

    EXPECT_DOUBLE_EQ(mean.compute(), 0.0);
}

TEST(MeanTest, ZeroWeightDoesNotAffectMean) {
    Mean mean;

    mean.update(4.0);
    mean.update(100.0, 0);

    EXPECT_DOUBLE_EQ(mean.compute(), 4.0);
}

TEST(MeanTest, OnlyZeroWeightObservations) {
    Mean mean;

    mean.update(10.0, 0);
    mean.update(20.0, 0);

    EXPECT_DOUBLE_EQ(mean.compute(), 0.0);
}

TEST(MeanTest, ZeroWeightInfinityDoesNotCorruptState) {
    Mean mean;

    mean.update(4.0);
    mean.update(std::numeric_limits<double>::infinity(), 0);

    EXPECT_DOUBLE_EQ(mean.compute(), 4.0);
}

TEST(MeanTest, ZeroWeightNaNDoesNotCorruptState) {
    Mean mean;

    mean.update(4.0);
    mean.update(std::numeric_limits<double>::quiet_NaN(), 0);

    EXPECT_DOUBLE_EQ(mean.compute(), 4.0);
}

TEST(MeanTest, ZeroWeightTensorDoesNotAffectMean) {
    Mean mean;

    mean.update(4.0);

    Tensor value({100.0f});
    mean.update(value, 0);

    EXPECT_DOUBLE_EQ(mean.compute(), 4.0);
}

TEST(MeanTest, ZeroWeightTensorInfinityDoesNotCorruptState) {
    Mean mean;

    mean.update(4.0);

    Tensor value({std::numeric_limits<float>::infinity()});
    mean.update(value, 0);

    EXPECT_DOUBLE_EQ(mean.compute(), 4.0);
}

// ==================== Input Validation ====================

TEST(MeanTest, RejectsVectorTensor) {
    Mean mean;

    Tensor value({1.0f, 2.0f, 3.0f});

    EXPECT_THROW(mean.update(value), std::invalid_argument);
}

TEST(MeanTest, RejectsMatrixTensor) {
    Mean mean;

    Tensor value({{1.0f, 2.0f}, {3.0f, 4.0f}});

    EXPECT_THROW(mean.update(value), std::invalid_argument);
}

TEST(MeanTest, RejectsNonScalarTensorWithZeroWeight) {
    Mean mean;

    Tensor value({1.0f, 2.0f});

    EXPECT_THROW(mean.update(value, 0), std::invalid_argument);
}

TEST(MeanTest, InvalidUpdatePreservesState) {
    Mean mean;

    mean.update(4.0, 3);

    Tensor invalid({1.0f, 2.0f});

    EXPECT_THROW(mean.update(invalid), std::invalid_argument);

    EXPECT_DOUBLE_EQ(mean.compute(), 4.0);

    mean.update(8.0);

    EXPECT_DOUBLE_EQ(mean.compute(), 5.0);
}

// ==================== State Management ====================

TEST(MeanTest, ResetClearsState) {
    Mean mean;

    mean.update(5.0, 10);
    mean.reset();

    EXPECT_DOUBLE_EQ(mean.compute(), 0.0);
}

TEST(MeanTest, UpdateAfterReset) {
    Mean mean;

    mean.update(100.0, 10);
    mean.reset();

    mean.update(2.0);
    mean.update(4.0);

    EXPECT_DOUBLE_EQ(mean.compute(), 3.0);
}

TEST(MeanTest, ComputeDoesNotResetState) {
    Mean mean;

    mean.update(2.0);

    EXPECT_DOUBLE_EQ(mean.compute(), 2.0);
    EXPECT_DOUBLE_EQ(mean.compute(), 2.0);

    mean.update(6.0);

    EXPECT_DOUBLE_EQ(mean.compute(), 4.0);
}

TEST(MeanTest, IndependentInstances) {
    Mean first;
    Mean second;

    first.update(2.0);
    second.update(8.0);

    EXPECT_DOUBLE_EQ(first.compute(), 2.0);
    EXPECT_DOUBLE_EQ(second.compute(), 8.0);

    first.reset();

    EXPECT_DOUBLE_EQ(first.compute(), 0.0);
    EXPECT_DOUBLE_EQ(second.compute(), 8.0);
}

// ==================== Autograd Integration ====================

TEST(MeanTest, DoesNotModifyGradients) {
    Mean mean;

    Tensor x({3.0f}, true);
    Tensor loss = x * x;

    loss.backward();

    af::array original_grad = x.grad().copy();

    mean.update(loss);

    EXPECT_DOUBLE_EQ(mean.compute(), 9.0);
    EXPECT_TRUE(x.requires_grad());
    EXPECT_TRUE(loss.requires_grad());

    EXPECT_TRUE(af::allTrue<bool>(x.grad() == original_grad));
}

TEST(MeanTest, StoresValueRatherThanTensor) {
    Mean mean;

    {
        Tensor value({5.0f}, true);
        mean.update(value);
    }

    EXPECT_DOUBLE_EQ(mean.compute(), 5.0);
}