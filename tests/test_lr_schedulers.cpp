#include "neuro/Tensor.hpp"
#include "neuro/optim/Adam.hpp"
#include "neuro/optim/CosineAnnealingLR.hpp"
#include "neuro/optim/ExponentialLR.hpp"
#include "neuro/optim/LRScheduler.hpp"
#include "neuro/optim/Optimizer.hpp"
#include "neuro/optim/SGD.hpp"
#include "neuro/optim/StepLR.hpp"

#include <arrayfire.h>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <vector>

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

TEST(OptimizerLearningRateTest, GetterAndSetter) {
    DummyOptimizer optimizer(0.01);

    EXPECT_DOUBLE_EQ(optimizer.learningRate(), 0.01);

    optimizer.setLearningRate(0.001);

    EXPECT_DOUBLE_EQ(optimizer.learningRate(), 0.001);
}

TEST(OptimizerLearningRateTest, AllowsZero) {
    DummyOptimizer optimizer(0.0);

    EXPECT_DOUBLE_EQ(optimizer.learningRate(), 0.0);

    optimizer.setLearningRate(0.1);
    optimizer.setLearningRate(0.0);

    EXPECT_DOUBLE_EQ(optimizer.learningRate(), 0.0);
}

TEST(OptimizerLearningRateTest, RejectsInvalidLearningRates) {
    DummyOptimizer optimizer(0.01);

    for (double lr : {-0.1, kInf, -kInf, kNaN}) {
        EXPECT_THROW(DummyOptimizer{lr}, std::invalid_argument);

        EXPECT_THROW(optimizer.setLearningRate(lr), std::invalid_argument);

        EXPECT_DOUBLE_EQ(optimizer.learningRate(), 0.01);
    }
}

TEST(OptimizerLearningRateTest, SGDUsesUpdatedLearningRate) {
    Tensor parameter(af::constant(2.0f, 1), true);

    SGD optimizer({parameter}, 0.1);

    optimizer.zeroGrad();
    parameter.sum().backward();
    optimizer.step();

    EXPECT_NEAR(parameter.data().scalar<float>(), 1.9, 1e-6);

    optimizer.setLearningRate(0.01);

    optimizer.zeroGrad();
    parameter.sum().backward();
    optimizer.step();

    EXPECT_NEAR(parameter.data().scalar<float>(), 1.89, 1e-6);
}

TEST(OptimizerLearningRateTest, ZeroLearningRatePreventsParameterUpdate) {
    Tensor parameter(af::constant(2.0f, 1), true);

    SGD optimizer({parameter}, 0.0);

    optimizer.zeroGrad();
    parameter.sum().backward();
    optimizer.step();

    EXPECT_FLOAT_EQ(parameter.data().scalar<float>(), 2.0f);
}

TEST(OptimizerLearningRateTest, AdamPreservesMomentsAfterLearningRateChange) {
    Tensor parameter(af::constant(2.0f, 1), true);

    constexpr double beta1 = 0.9;
    constexpr double beta2 = 0.999;
    constexpr double eps = 1e-8;

    Adam optimizer({parameter}, 0.1);

    const std::vector<double> gradients{2.0, -1.0, 3.0, -2.0};
    const std::vector<double> learning_rates{0.1, 0.01, 0.0, 0.05};

    double expected = 2.0;
    double m = 0.0;
    double v = 0.0;

    for (std::size_t i = 0; i < gradients.size(); ++i) {
        const double g = gradients[i];
        const double lr = learning_rates[i];
        const double t = static_cast<double>(i + 1);

        optimizer.setLearningRate(lr);
        optimizer.zeroGrad();

        Tensor loss = (parameter * g).sum();
        loss.backward();
        optimizer.step();

        m = beta1 * m + (1.0 - beta1) * g;
        v = beta2 * v + (1.0 - beta2) * g * g;

        const double m_hat = m / (1.0 - std::pow(beta1, t));
        const double v_hat = v / (1.0 - std::pow(beta2, t));

        expected -= lr * m_hat / (std::sqrt(v_hat) + eps);

        EXPECT_NEAR(parameter.data().scalar<float>(), expected, 1e-5) << "Step " << i + 1;
    }
}

TEST(StepLRTest, DoesNotChangeLearningRateOnConstruction) {
    DummyOptimizer optimizer(0.01);

    StepLR scheduler(optimizer, 3, 0.5);

    EXPECT_DOUBLE_EQ(optimizer.learningRate(), 0.01);
}

TEST(StepLRTest, DecaysAtCorrectIntervals) {
    DummyOptimizer optimizer(0.01);
    StepLR scheduler(optimizer, 3, 0.5);

    const std::vector<double> expected{0.01,   0.01,   0.005,  0.005,  0.005,
                                       0.0025, 0.0025, 0.0025, 0.00125};

    for (std::size_t i = 0; i < expected.size(); ++i) {
        scheduler.step();

        EXPECT_NEAR(optimizer.learningRate(), expected[i], kTolerance) << "Epoch " << i + 1;
    }
}

TEST(StepLRTest, StepSizeOneDecaysEveryEpoch) {
    DummyOptimizer optimizer(0.01);
    StepLR scheduler(optimizer, 1, 0.5);

    scheduler.step();
    EXPECT_NEAR(optimizer.learningRate(), 0.005, kTolerance);

    scheduler.step();
    EXPECT_NEAR(optimizer.learningRate(), 0.0025, kTolerance);
}

TEST(StepLRTest, UsesCurrentLearningRate) {
    DummyOptimizer optimizer(0.01);
    StepLR scheduler(optimizer, 2, 0.5);

    scheduler.step();

    optimizer.setLearningRate(0.02);

    scheduler.step();

    EXPECT_NEAR(optimizer.learningRate(), 0.01, kTolerance);
}

TEST(StepLRTest, SupportsIncreasingLearningRate) {
    DummyOptimizer optimizer(0.01);
    StepLR scheduler(optimizer, 2, 2.0);

    scheduler.step();
    EXPECT_DOUBLE_EQ(optimizer.learningRate(), 0.01);

    scheduler.step();
    EXPECT_DOUBLE_EQ(optimizer.learningRate(), 0.02);
}

TEST(StepLRTest, RejectsInvalidArguments) {
    DummyOptimizer optimizer(0.01);

    EXPECT_THROW(StepLR(optimizer, 0), std::invalid_argument);

    for (double gamma : {0.0, -0.5, kInf, -kInf, kNaN}) {
        EXPECT_THROW(StepLR(optimizer, 3, gamma), std::invalid_argument);
    }
}

TEST(ExponentialLRTest, MatchesClosedFormSchedule) {
    constexpr double initial_lr = 0.01;
    constexpr double gamma = 0.95;

    DummyOptimizer optimizer(initial_lr);
    ExponentialLR scheduler(optimizer, gamma);

    EXPECT_DOUBLE_EQ(optimizer.learningRate(), initial_lr);

    for (int epoch = 1; epoch <= 30; ++epoch) {
        scheduler.step();

        EXPECT_NEAR(optimizer.learningRate(), initial_lr * std::pow(gamma, epoch), kTolerance)
            << "Epoch " << epoch;
    }
}

TEST(ExponentialLRTest, GammaOnePreservesLearningRate) {
    DummyOptimizer optimizer(0.01);
    ExponentialLR scheduler(optimizer, 1.0);

    for (int i = 0; i < 10; ++i) {
        scheduler.step();
    }

    EXPECT_DOUBLE_EQ(optimizer.learningRate(), 0.01);
}

TEST(ExponentialLRTest, SupportsIncreasingLearningRate) {
    DummyOptimizer optimizer(0.01);
    ExponentialLR scheduler(optimizer, 2.0);

    scheduler.step();
    EXPECT_DOUBLE_EQ(optimizer.learningRate(), 0.02);

    scheduler.step();
    EXPECT_DOUBLE_EQ(optimizer.learningRate(), 0.04);
}

TEST(ExponentialLRTest, UsesCurrentLearningRate) {
    DummyOptimizer optimizer(0.01);
    ExponentialLR scheduler(optimizer, 0.5);

    scheduler.step();

    optimizer.setLearningRate(0.04);

    scheduler.step();

    EXPECT_NEAR(optimizer.learningRate(), 0.02, kTolerance);
}

TEST(ExponentialLRTest, RejectsInvalidGamma) {
    DummyOptimizer optimizer(0.01);

    for (double gamma : {0.0, -0.5, kInf, -kInf, kNaN}) {
        EXPECT_THROW(ExponentialLR(optimizer, gamma), std::invalid_argument);
    }
}

TEST(CosineAnnealingLRTest, MatchesKnownSchedule) {
    DummyOptimizer optimizer(0.01);
    CosineAnnealingLR scheduler(optimizer, 4, 0.002);

    const std::vector<double> expected{0.00882842712474619, 0.006, 0.00317157287525381, 0.002};

    EXPECT_DOUBLE_EQ(optimizer.learningRate(), 0.01);

    for (std::size_t i = 0; i < expected.size(); ++i) {
        scheduler.step();

        EXPECT_NEAR(optimizer.learningRate(), expected[i], kTolerance) << "Epoch " << i + 1;
    }
}

TEST(CosineAnnealingLRTest, ReachesZeroWithDefaultMinimum) {
    DummyOptimizer optimizer(0.01);
    CosineAnnealingLR scheduler(optimizer, 10);

    for (int i = 0; i < 10; ++i) {
        scheduler.step();
    }

    EXPECT_NEAR(optimizer.learningRate(), 0.0, kTolerance);
}

TEST(CosineAnnealingLRTest, RemainsAtMinimumAfterCompletion) {
    DummyOptimizer optimizer(0.01);
    CosineAnnealingLR scheduler(optimizer, 4, 0.001);

    for (int i = 0; i < 4; ++i) {
        scheduler.step();
    }

    EXPECT_NEAR(optimizer.learningRate(), 0.001, kTolerance);

    for (int i = 0; i < 20; ++i) {
        scheduler.step();

        EXPECT_NEAR(optimizer.learningRate(), 0.001, kTolerance);
    }
}

TEST(CosineAnnealingLRTest, DecreasesMonotonically) {
    DummyOptimizer optimizer(0.01);
    CosineAnnealingLR scheduler(optimizer, 30, 0.001);

    double previous = optimizer.learningRate();

    for (int i = 0; i < 30; ++i) {
        scheduler.step();

        const double current = optimizer.learningRate();

        EXPECT_LE(current, previous);
        EXPECT_GE(current, 0.001);

        previous = current;
    }
}

TEST(CosineAnnealingLRTest, SingleEpochSchedule) {
    DummyOptimizer optimizer(0.01);
    CosineAnnealingLR scheduler(optimizer, 1, 0.001);

    scheduler.step();

    EXPECT_NEAR(optimizer.learningRate(), 0.001, kTolerance);
}

TEST(CosineAnnealingLRTest, EqualBaseAndMinimumLearningRates) {
    DummyOptimizer optimizer(0.01);
    CosineAnnealingLR scheduler(optimizer, 10, 0.01);

    for (int i = 0; i < 15; ++i) {
        scheduler.step();

        EXPECT_DOUBLE_EQ(optimizer.learningRate(), 0.01);
    }
}

TEST(CosineAnnealingLRTest, UsesOriginalBaseLearningRate) {
    DummyOptimizer optimizer(0.01);
    CosineAnnealingLR scheduler(optimizer, 4, 0.002);

    optimizer.setLearningRate(0.05);

    scheduler.step();

    EXPECT_NEAR(optimizer.learningRate(), 0.00882842712474619, kTolerance);
}

TEST(CosineAnnealingLRTest, RejectsInvalidArguments) {
    DummyOptimizer optimizer(0.01);

    EXPECT_THROW(CosineAnnealingLR(optimizer, 0), std::invalid_argument);

    for (double eta_min : {-0.001, 0.02, kInf, -kInf, kNaN}) {
        EXPECT_THROW(CosineAnnealingLR(optimizer, 10, eta_min), std::invalid_argument);
    }
}

TEST(LRSchedulerTest, SupportsPolymorphicStepping) {
    DummyOptimizer optimizer(0.01);

    StepLR step(optimizer, 1, 0.5);
    ExponentialLR exponential(optimizer, 0.5);
    CosineAnnealingLR cosine(optimizer, 1, 0.001);

    std::vector<LRScheduler*> schedulers{&step, &exponential, &cosine};

    schedulers[0]->step();
    EXPECT_NEAR(optimizer.learningRate(), 0.005, kTolerance);

    schedulers[1]->step();
    EXPECT_NEAR(optimizer.learningRate(), 0.0025, kTolerance);

    schedulers[2]->step();
    EXPECT_NEAR(optimizer.learningRate(), 0.001, kTolerance);
}

TEST(LRSchedulerTest, SchedulerAffectsActualOptimizerUpdates) {
    Tensor parameter(af::constant(2.0f, 1), true);

    SGD optimizer({parameter}, 0.1);
    StepLR scheduler(optimizer, 1, 0.5);

    const std::vector<double> expected{1.9, 1.85, 1.825, 1.8125};

    for (std::size_t i = 0; i < expected.size(); ++i) {
        optimizer.zeroGrad();
        parameter.sum().backward();
        optimizer.step();

        scheduler.step();

        EXPECT_NEAR(parameter.data().scalar<float>(), expected[i], 1e-6) << "Epoch " << i + 1;
    }
}