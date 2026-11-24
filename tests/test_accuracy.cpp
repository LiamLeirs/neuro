#include "neuro/Tensor.hpp"
#include "neuro/metrics/Accuracy.hpp"

#include <af/data.h>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

using namespace neuro;
using namespace neuro::metrics;

namespace {

Tensor makeTargets(const std::vector<int>& labels) {
    return Tensor(af::array(labels.size(), labels.data()));
}

} // namespace

// ==================== Basic Accuracy ====================

TEST(AccuracyTest, PerfectAccuracy) {
    Accuracy accuracy;

    Tensor predictions({{0.1f, 0.8f, 0.2f}, {0.9f, 0.3f, 0.1f}, {0.2f, 0.1f, 0.9f}});

    accuracy.update(predictions, makeTargets({1, 0, 2}));

    EXPECT_DOUBLE_EQ(accuracy.compute(), 1.0);
}

TEST(AccuracyTest, ZeroAccuracy) {
    Accuracy accuracy;

    Tensor predictions({{0.1f, 0.9f}, {0.8f, 0.2f}});

    accuracy.update(predictions, makeTargets({0, 1}));

    EXPECT_DOUBLE_EQ(accuracy.compute(), 0.0);
}

TEST(AccuracyTest, PartialAccuracy) {
    Accuracy accuracy;

    Tensor predictions({{0.9f, 0.1f}, {0.1f, 0.9f}, {0.8f, 0.2f}, {0.2f, 0.8f}});

    accuracy.update(predictions, makeTargets({0, 1, 1, 1}));

    EXPECT_DOUBLE_EQ(accuracy.compute(), 0.75);
}

TEST(AccuracyTest, NegativeLogits) {
    Accuracy accuracy;

    Tensor predictions({{-5.0f, -1.0f, -3.0f}, {-2.0f, -8.0f, -4.0f}});

    accuracy.update(predictions, makeTargets({1, 0}));

    EXPECT_DOUBLE_EQ(accuracy.compute(), 1.0);
}

TEST(AccuracyTest, SingleSample) {
    Accuracy accuracy;

    Tensor predictions(af::array(af::constant(0.0f, 1, 3)));

    const float logits[] = {0.1f, 0.2f, 0.9f};
    predictions = Tensor(af::array(1, 3, logits));

    accuracy.update(predictions, makeTargets({2}));

    EXPECT_DOUBLE_EQ(accuracy.compute(), 1.0);
}

TEST(AccuracyTest, UnsignedByteTargets) {
    Accuracy accuracy;

    Tensor predictions({{0.1f, 0.9f}, {0.8f, 0.2f}});

    const unsigned char labels[] = {1, 0};
    Tensor targets(af::array(2, labels));

    accuracy.update(predictions, targets);

    EXPECT_DOUBLE_EQ(accuracy.compute(), 1.0);
}

// ==================== Batch Accumulation ====================

TEST(AccuracyTest, MultipleBatches) {
    Accuracy accuracy;

    Tensor predictions({{0.9f, 0.1f}, {0.1f, 0.9f}});

    accuracy.update(predictions, makeTargets({0, 1}));
    accuracy.update(predictions, makeTargets({1, 0}));

    EXPECT_DOUBLE_EQ(accuracy.compute(), 0.5);
}

TEST(AccuracyTest, UnequalBatchSizes) {
    Accuracy accuracy;

    Tensor first_batch({{0.9f, 0.1f}, {0.1f, 0.9f}, {0.8f, 0.2f}});

    const float logits[] = {0.9f, 0.1f};
    Tensor second_batch(af::array(1, 2, logits));

    // Three correct predictions out of three.
    accuracy.update(first_batch, makeTargets({0, 1, 0}));

    // Zero correct predictions out of one.
    accuracy.update(second_batch, makeTargets({1}));

    // Must be 3/4, not the mean of the two batch accuracies.
    EXPECT_DOUBLE_EQ(accuracy.compute(), 0.75);
}

TEST(AccuracyTest, ComputeDoesNotResetState) {
    Accuracy accuracy;

    Tensor predictions({{0.9f, 0.1f}, {0.1f, 0.9f}});

    accuracy.update(predictions, makeTargets({0, 1}));

    EXPECT_DOUBLE_EQ(accuracy.compute(), 1.0);
    EXPECT_DOUBLE_EQ(accuracy.compute(), 1.0);

    accuracy.update(predictions, makeTargets({1, 0}));

    EXPECT_DOUBLE_EQ(accuracy.compute(), 0.5);
}

// ==================== State Management ====================

TEST(AccuracyTest, EmptyStateReturnsZero) {
    Accuracy accuracy;

    EXPECT_DOUBLE_EQ(accuracy.compute(), 0.0);
}

TEST(AccuracyTest, ResetClearsState) {
    Accuracy accuracy;

    Tensor predictions({{0.9f, 0.1f}, {0.1f, 0.9f}});

    accuracy.update(predictions, makeTargets({0, 1}));

    EXPECT_DOUBLE_EQ(accuracy.compute(), 1.0);

    accuracy.reset();

    EXPECT_DOUBLE_EQ(accuracy.compute(), 0.0);
}

TEST(AccuracyTest, UpdateAfterReset) {
    Accuracy accuracy;

    Tensor predictions({{0.9f, 0.1f}, {0.1f, 0.9f}});

    accuracy.update(predictions, makeTargets({0, 1}));
    accuracy.reset();

    accuracy.update(predictions, makeTargets({0, 0}));

    EXPECT_DOUBLE_EQ(accuracy.compute(), 0.5);
}

TEST(AccuracyTest, IndependentInstances) {
    Accuracy first;
    Accuracy second;

    Tensor predictions({{0.9f, 0.1f}, {0.1f, 0.9f}});

    first.update(predictions, makeTargets({0, 1}));
    second.update(predictions, makeTargets({1, 0}));

    EXPECT_DOUBLE_EQ(first.compute(), 1.0);
    EXPECT_DOUBLE_EQ(second.compute(), 0.0);
}

// ==================== Input Validation ====================

TEST(AccuracyTest, RejectsMismatchedBatchSizes) {
    Accuracy accuracy;

    Tensor predictions({{0.9f, 0.1f}, {0.1f, 0.9f}});

    EXPECT_THROW(accuracy.update(predictions, makeTargets({0})), std::invalid_argument);
}

TEST(AccuracyTest, RejectsFloatingPointTargets) {
    Accuracy accuracy;

    Tensor predictions({{0.9f, 0.1f}, {0.1f, 0.9f}});

    Tensor targets({0.0f, 1.0f});

    EXPECT_THROW(accuracy.update(predictions, targets), std::invalid_argument);
}

TEST(AccuracyTest, RejectsNegativeClassIndex) {
    Accuracy accuracy;

    Tensor predictions({{0.9f, 0.1f}, {0.1f, 0.9f}});

    EXPECT_THROW(accuracy.update(predictions, makeTargets({-1, 1})), std::invalid_argument);
}

TEST(AccuracyTest, RejectsClassIndexOutOfBounds) {
    Accuracy accuracy;

    Tensor predictions({{0.9f, 0.1f}, {0.1f, 0.9f}});

    EXPECT_THROW(accuracy.update(predictions, makeTargets({0, 2})), std::invalid_argument);
}

TEST(AccuracyTest, RejectsThreeDimensionalPredictions) {
    Accuracy accuracy;

    Tensor predictions(af::constant(1.0f, 2, 3, 2));

    EXPECT_THROW(accuracy.update(predictions, makeTargets({0, 1})), std::invalid_argument);
}

TEST(AccuracyTest, RejectsFourDimensionalPredictions) {
    Accuracy accuracy;

    Tensor predictions(af::constant(1.0f, 2, 3, 1, 2));

    EXPECT_THROW(accuracy.update(predictions, makeTargets({0, 1})), std::invalid_argument);
}

TEST(AccuracyTest, RejectsMultipleTargetColumns) {
    Accuracy accuracy;

    Tensor predictions({{0.9f, 0.1f}, {0.1f, 0.9f}});

    Tensor targets(af::constant(0, 2, 2, s32));

    EXPECT_THROW(accuracy.update(predictions, targets), std::invalid_argument);
}

TEST(AccuracyTest, RejectsThreeDimensionalTargets) {
    Accuracy accuracy;

    Tensor predictions({{0.9f, 0.1f}, {0.1f, 0.9f}});

    Tensor targets(af::constant(0, 2, 1, 2, s32));

    EXPECT_THROW(accuracy.update(predictions, targets), std::invalid_argument);
}

TEST(AccuracyTest, InvalidUpdatePreservesState) {
    Accuracy accuracy;

    Tensor predictions({{0.9f, 0.1f}, {0.1f, 0.9f}});

    accuracy.update(predictions, makeTargets({0, 1}));

    EXPECT_THROW(accuracy.update(predictions, makeTargets({0, 5})), std::invalid_argument);

    EXPECT_DOUBLE_EQ(accuracy.compute(), 1.0);

    accuracy.update(predictions, makeTargets({1, 0}));

    EXPECT_DOUBLE_EQ(accuracy.compute(), 0.5);
}

// ==================== Autograd Integration ====================

TEST(AccuracyTest, DoesNotModifyGradients) {
    Accuracy accuracy;

    Tensor predictions({{0.9f, 0.1f}, {0.1f, 0.9f}}, true);

    predictions.sum().backward();

    af::array original_grad = predictions.grad().copy();

    accuracy.update(predictions, makeTargets({0, 1}));

    EXPECT_TRUE(predictions.requires_grad());
    EXPECT_TRUE(af::allTrue<bool>(predictions.grad() == original_grad));

    EXPECT_DOUBLE_EQ(accuracy.compute(), 1.0);
}