#include "neuro/optim/Optimizer.hpp"
#include "neuro/optim/ReduceLROnPlateau.hpp"

#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

using namespace neuro;

namespace {

class DummyOptimizer : public Optimizer {
public:
    explicit DummyOptimizer(double lr) : Optimizer({}, lr) {}

    void step() override {}
};

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kTolerance = 1e-12;

} // namespace

TEST(ReduceLROnPlateauTest, DoesNotChangeLearningRateOnConstruction) {
    DummyOptimizer optimizer(0.01);
    ReduceLROnPlateau scheduler(optimizer);

    EXPECT_DOUBLE_EQ(optimizer.learningRate(), 0.01);
}

TEST(ReduceLROnPlateauTest, RejectsInvalidMode) {
    DummyOptimizer optimizer(0.01);

    EXPECT_THROW(ReduceLROnPlateau(optimizer, static_cast<PlateauMode>(99)), std::invalid_argument);
}

TEST(ReduceLROnPlateauTest, RejectsInvalidFactor) {
    DummyOptimizer optimizer(0.01);

    for (double factor : {0.0, -0.5, 1.0, 1.5, kInf, -kInf, kNaN}) {
        EXPECT_THROW(ReduceLROnPlateau(optimizer, PlateauMode::Min, factor), std::invalid_argument);
    }
}

TEST(ReduceLROnPlateauTest, RejectsInvalidThreshold) {
    DummyOptimizer optimizer(0.01);

    for (double threshold : {-0.1, kInf, -kInf, kNaN}) {
        EXPECT_THROW(ReduceLROnPlateau(optimizer, PlateauMode::Min, 0.5, 2, threshold),
                     std::invalid_argument);
    }
}

TEST(ReduceLROnPlateauTest, RejectsInvalidMinimumLearningRate) {
    DummyOptimizer optimizer(0.01);

    for (double min_lr : {-0.1, 0.02, kInf, -kInf, kNaN}) {
        EXPECT_THROW(ReduceLROnPlateau(optimizer, PlateauMode::Min, 0.5, 2, 0.0, min_lr),
                     std::invalid_argument);
    }
}

TEST(ReduceLROnPlateauTest, FirstObservationEstablishesBaseline) {
    DummyOptimizer optimizer(0.01);

    ReduceLROnPlateau scheduler(optimizer, PlateauMode::Min, 0.5, 0);

    scheduler.step(10.0);

    EXPECT_DOUBLE_EQ(optimizer.learningRate(), 0.01);
}

TEST(ReduceLROnPlateauTest, ImprovingLossPreservesLearningRate) {
    DummyOptimizer optimizer(0.01);

    ReduceLROnPlateau scheduler(optimizer, PlateauMode::Min, 0.5, 0);

    for (double loss : {1.0, 0.8, 0.6, 0.4, 0.2}) {
        scheduler.step(loss);

        EXPECT_DOUBLE_EQ(optimizer.learningRate(), 0.01);
    }
}

TEST(ReduceLROnPlateauTest, ReducesAfterPatienceIsExceeded) {
    DummyOptimizer optimizer(0.01);

    ReduceLROnPlateau scheduler(optimizer, PlateauMode::Min, 0.5, 2);

    scheduler.step(0.5);

    scheduler.step(0.6);
    EXPECT_DOUBLE_EQ(optimizer.learningRate(), 0.01);

    scheduler.step(0.7);
    EXPECT_DOUBLE_EQ(optimizer.learningRate(), 0.01);

    scheduler.step(0.8);
    EXPECT_NEAR(optimizer.learningRate(), 0.005, kTolerance);
}

TEST(ReduceLROnPlateauTest, ZeroPatienceReducesImmediately) {
    DummyOptimizer optimizer(0.01);

    ReduceLROnPlateau scheduler(optimizer, PlateauMode::Min, 0.5, 0);

    scheduler.step(0.5);
    EXPECT_DOUBLE_EQ(optimizer.learningRate(), 0.01);

    scheduler.step(0.6);
    EXPECT_NEAR(optimizer.learningRate(), 0.005, kTolerance);

    scheduler.step(0.7);
    EXPECT_NEAR(optimizer.learningRate(), 0.0025, kTolerance);
}

TEST(ReduceLROnPlateauTest, ImprovementResetsBadEpochs) {
    DummyOptimizer optimizer(0.01);

    ReduceLROnPlateau scheduler(optimizer, PlateauMode::Min, 0.5, 2);

    scheduler.step(0.5);
    scheduler.step(0.6);
    scheduler.step(0.7);

    scheduler.step(0.4);

    scheduler.step(0.5);
    scheduler.step(0.6);

    EXPECT_DOUBLE_EQ(optimizer.learningRate(), 0.01);

    scheduler.step(0.7);

    EXPECT_NEAR(optimizer.learningRate(), 0.005, kTolerance);
}

TEST(ReduceLROnPlateauTest, MaximizationMode) {
    DummyOptimizer optimizer(0.01);

    ReduceLROnPlateau scheduler(optimizer, PlateauMode::Max, 0.5, 1);

    scheduler.step(0.5);
    scheduler.step(0.6);
    scheduler.step(0.7);

    EXPECT_DOUBLE_EQ(optimizer.learningRate(), 0.01);

    scheduler.step(0.65);
    EXPECT_DOUBLE_EQ(optimizer.learningRate(), 0.01);

    scheduler.step(0.64);
    EXPECT_NEAR(optimizer.learningRate(), 0.005, kTolerance);
}

TEST(ReduceLROnPlateauTest, SupportsNegativeMetrics) {
    DummyOptimizer optimizer(0.01);

    ReduceLROnPlateau scheduler(optimizer, PlateauMode::Max, 0.5, 0);

    scheduler.step(-10.0);
    scheduler.step(-5.0);

    EXPECT_DOUBLE_EQ(optimizer.learningRate(), 0.01);

    scheduler.step(-6.0);

    EXPECT_NEAR(optimizer.learningRate(), 0.005, kTolerance);
}

TEST(ReduceLROnPlateauTest, MinimumModeUsesAbsoluteThreshold) {
    DummyOptimizer optimizer(0.01);

    ReduceLROnPlateau scheduler(optimizer, PlateauMode::Min, 0.5, 0, 0.01);

    scheduler.step(0.5);

    // Improvement exceeds the threshold.
    scheduler.step(0.48);
    EXPECT_DOUBLE_EQ(optimizer.learningRate(), 0.01);

    // Improvement is smaller than the threshold.
    scheduler.step(0.475);
    EXPECT_NEAR(optimizer.learningRate(), 0.005, kTolerance);
}

TEST(ReduceLROnPlateauTest, MaximumModeUsesAbsoluteThreshold) {
    DummyOptimizer optimizer(0.01);

    ReduceLROnPlateau scheduler(optimizer, PlateauMode::Max, 0.5, 0, 0.01);

    scheduler.step(0.5);

    scheduler.step(0.52);
    EXPECT_DOUBLE_EQ(optimizer.learningRate(), 0.01);

    scheduler.step(0.525);
    EXPECT_NEAR(optimizer.learningRate(), 0.005, kTolerance);
}

TEST(ReduceLROnPlateauTest, ImprovementEqualToThresholdIsInsufficient) {
    for (auto mode : {PlateauMode::Min, PlateauMode::Max}) {
        DummyOptimizer optimizer(0.01);

        ReduceLROnPlateau scheduler(optimizer, mode, 0.5, 0, 0.125);

        scheduler.step(0.5);

        scheduler.step(mode == PlateauMode::Min ? 0.375 : 0.625);

        EXPECT_NEAR(optimizer.learningRate(), 0.005, kTolerance);
    }
}

TEST(ReduceLROnPlateauTest, EqualMetricDoesNotCountAsImprovement) {
    DummyOptimizer optimizer(0.01);

    ReduceLROnPlateau scheduler(optimizer, PlateauMode::Min, 0.5, 0, 0.0);

    scheduler.step(0.5);
    scheduler.step(0.5);

    EXPECT_NEAR(optimizer.learningRate(), 0.005, kTolerance);
}

TEST(ReduceLROnPlateauTest, SupportsRepeatedReductions) {
    DummyOptimizer optimizer(0.01);

    ReduceLROnPlateau scheduler(optimizer, PlateauMode::Min, 0.5, 1);

    scheduler.step(0.5);

    double expected = 0.01;

    for (int i = 0; i < 4; ++i) {
        scheduler.step(0.6);
        EXPECT_NEAR(optimizer.learningRate(), expected, kTolerance);

        scheduler.step(0.6);
        expected *= 0.5;

        EXPECT_NEAR(optimizer.learningRate(), expected, kTolerance);
    }
}

TEST(ReduceLROnPlateauTest, PreservesHistoricalBestAfterReduction) {
    DummyOptimizer optimizer(0.01);

    ReduceLROnPlateau scheduler(optimizer, PlateauMode::Min, 0.5, 0);

    scheduler.step(0.5);
    scheduler.step(0.8);

    EXPECT_NEAR(optimizer.learningRate(), 0.005, kTolerance);

    // Better than the previous epoch, but not the historical best.
    scheduler.step(0.7);

    EXPECT_NEAR(optimizer.learningRate(), 0.0025, kTolerance);

    // A genuine improvement.
    scheduler.step(0.4);

    EXPECT_NEAR(optimizer.learningRate(), 0.0025, kTolerance);
}

TEST(ReduceLROnPlateauTest, RespectsMinimumLearningRate) {
    DummyOptimizer optimizer(0.01);

    ReduceLROnPlateau scheduler(optimizer, PlateauMode::Min, 0.5, 0, 0.0, 0.003);

    scheduler.step(0.5);

    scheduler.step(0.6);
    EXPECT_NEAR(optimizer.learningRate(), 0.005, kTolerance);

    scheduler.step(0.6);
    EXPECT_NEAR(optimizer.learningRate(), 0.003, kTolerance);

    for (int i = 0; i < 10; ++i) {
        scheduler.step(0.6);

        EXPECT_NEAR(optimizer.learningRate(), 0.003, kTolerance);
    }
}

TEST(ReduceLROnPlateauTest, NeverIncreasesLearningRate) {
    DummyOptimizer optimizer(0.01);

    ReduceLROnPlateau scheduler(optimizer, PlateauMode::Min, 0.5, 0, 0.0, 0.003);

    scheduler.step(0.5);

    optimizer.setLearningRate(0.001);

    scheduler.step(0.6);

    EXPECT_DOUBLE_EQ(optimizer.learningRate(), 0.001);
}

TEST(ReduceLROnPlateauTest, UsesCurrentLearningRate) {
    DummyOptimizer optimizer(0.01);

    ReduceLROnPlateau scheduler(optimizer, PlateauMode::Min, 0.5, 0);

    scheduler.step(0.5);

    optimizer.setLearningRate(0.02);

    scheduler.step(0.6);

    EXPECT_NEAR(optimizer.learningRate(), 0.01, kTolerance);
}

TEST(ReduceLROnPlateauTest, SupportsZeroLearningRate) {
    DummyOptimizer optimizer(0.0);

    ReduceLROnPlateau scheduler(optimizer, PlateauMode::Min, 0.5, 0);

    scheduler.step(0.5);
    scheduler.step(0.6);

    EXPECT_DOUBLE_EQ(optimizer.learningRate(), 0.0);
}

TEST(ReduceLROnPlateauTest, RejectsNonFiniteMetricsWithoutChangingState) {
    DummyOptimizer optimizer(0.01);

    ReduceLROnPlateau scheduler(optimizer, PlateauMode::Min, 0.5, 1);

    scheduler.step(0.5);

    for (double metric : {kNaN, kInf, -kInf}) {
        EXPECT_THROW(scheduler.step(metric), std::invalid_argument);

        EXPECT_DOUBLE_EQ(optimizer.learningRate(), 0.01);
    }

    scheduler.step(0.6);
    EXPECT_DOUBLE_EQ(optimizer.learningRate(), 0.01);

    scheduler.step(0.7);
    EXPECT_NEAR(optimizer.learningRate(), 0.005, kTolerance);
}