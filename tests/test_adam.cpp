
#include "neuro/Tensor.hpp"
#include "neuro/optim/Adam.hpp"
#include "neuro/optim/Optimizer.hpp"

#include <arrayfire.h>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace neuro;

namespace {

constexpr double kTolerance = 1e-5;

// Reference implementation using ordinary CPU doubles.
// Deliberately independent of Neuro and ArrayFire.
struct ReferenceAdam {
    double value;
    double m = 0.0;
    double v = 0.0;
    int t = 0;

    double lr;
    double b1 = 0.9;
    double b2 = 0.999;
    double eps = 1e-8;

    void step(double gradient) {
        ++t;

        m = b1 * m + (1.0 - b1) * gradient;
        v = b2 * v + (1.0 - b2) * gradient * gradient;

        const double correctedM = m / (1.0 - std::pow(b1, t));
        const double correctedV = v / (1.0 - std::pow(b2, t));

        value -= lr * correctedM / (std::sqrt(correctedV) + eps);
    }
};

double scalarValue(const Tensor& tensor) {
    return tensor.data().scalar<float>();
}

// Produce an exact gradient through the autograd engine.
void backwardWithGradient(Tensor& parameter, double gradient) {
    Tensor loss = (parameter * gradient).sum();
    loss.backward();
}

} // namespace

TEST(AdamTest, SingleParameterUpdate) {
    Tensor parameter({1.0f}, true);
    Adam optimizer({parameter}, 0.1);

    backwardWithGradient(parameter, 0.5);
    optimizer.step();

    EXPECT_NEAR(scalarValue(parameter), 0.9, kTolerance);
}

TEST(AdamTest, MultipleStepsPreserveMomentState) {
    Tensor parameter({1.0f}, true);
    Adam optimizer({parameter}, 0.1);

    ReferenceAdam reference{.value = 1.0, .lr = 0.1};

    const std::vector<double> gradients = {0.5, -0.25, 1.0, 0.1, -0.75};

    for (double gradient : gradients) {
        optimizer.zeroGrad();

        backwardWithGradient(parameter, gradient);
        optimizer.step();

        reference.step(gradient);

        EXPECT_NEAR(scalarValue(parameter), reference.value, kTolerance);
    }
}

TEST(AdamTest, IndependentParameterStates) {
    Tensor first({1.0f}, true);
    Tensor second({2.0f}, true);

    Adam optimizer({first, second}, 0.05);

    ReferenceAdam firstReference{.value = 1.0, .lr = 0.05};
    ReferenceAdam secondReference{.value = 2.0, .lr = 0.05};

    const std::vector<double> firstGradients = {0.5, 0.25, -0.75, 1.0};

    const std::vector<double> secondGradients = {-1.0, -0.5, 0.25, -0.25};

    for (std::size_t i = 0; i < firstGradients.size(); ++i) {
        optimizer.zeroGrad();

        backwardWithGradient(first, firstGradients[i]);
        backwardWithGradient(second, secondGradients[i]);

        optimizer.step();

        firstReference.step(firstGradients[i]);
        secondReference.step(secondGradients[i]);

        EXPECT_NEAR(scalarValue(first), firstReference.value, kTolerance);

        EXPECT_NEAR(scalarValue(second), secondReference.value, kTolerance);
    }
}

TEST(AdamTest, UpdatesEveryElementIndependently) {
    Tensor parameter({1.0f, 2.0f, 3.0f}, true);
    Adam optimizer({parameter}, 0.1);

    // Gradient = [1, -1, 2]
    Tensor coefficients({1.0f, -1.0f, 2.0f});
    Tensor loss = (parameter * coefficients).sum();

    loss.backward();
    optimizer.step();

    const af::array actual = parameter.data();

    EXPECT_NEAR(actual(0).scalar<float>(), 0.9f, kTolerance);
    EXPECT_NEAR(actual(1).scalar<float>(), 2.1f, kTolerance);
    EXPECT_NEAR(actual(2).scalar<float>(), 2.9f, kTolerance);
}

TEST(AdamTest, ZeroGradClearsGradients) {
    Tensor parameter({1.0f}, true);
    Adam optimizer({parameter}, 0.1);

    backwardWithGradient(parameter, 0.5);

    EXPECT_NEAR(parameter.grad().scalar<float>(), 0.5f, kTolerance);

    optimizer.zeroGrad();

    EXPECT_NEAR(parameter.grad().scalar<float>(), 0.0f, kTolerance);
}

TEST(AdamTest, ZeroGradDoesNotResetOptimizerState) {
    Tensor parameter({1.0f}, true);
    Adam optimizer({parameter}, 0.1);

    ReferenceAdam reference{.value = 1.0, .lr = 0.1};

    backwardWithGradient(parameter, 0.5);
    optimizer.step();
    reference.step(0.5);

    optimizer.zeroGrad();

    backwardWithGradient(parameter, -0.25);
    optimizer.step();
    reference.step(-0.25);

    EXPECT_NEAR(scalarValue(parameter), reference.value, kTolerance);
}

TEST(AdamTest, IgnoresParametersWithoutGradientTracking) {
    Tensor frozen({1.0f}, false);
    Tensor trainable({2.0f}, true);

    Adam optimizer({frozen, trainable}, 0.1);

    backwardWithGradient(trainable, 0.5);
    optimizer.step();

    EXPECT_NEAR(scalarValue(frozen), 1.0f, kTolerance);
    EXPECT_NEAR(scalarValue(trainable), 1.9f, kTolerance);
}

TEST(AdamTest, RejectsInvalidLearningRate) {
    Tensor parameter({1.0f}, true);

    EXPECT_THROW(Adam({parameter}, 0.0), std::invalid_argument);
    EXPECT_THROW(Adam({parameter}, -0.1), std::invalid_argument);
}

TEST(AdamTest, RejectsInvalidBetaValues) {
    Tensor parameter({1.0f}, true);

    EXPECT_THROW(Adam({parameter}, 0.01, -0.1, 0.999), std::invalid_argument);
    EXPECT_THROW(Adam({parameter}, 0.01, 1.0, 0.999), std::invalid_argument);

    EXPECT_THROW(Adam({parameter}, 0.01, 0.9, -0.1), std::invalid_argument);
    EXPECT_THROW(Adam({parameter}, 0.01, 0.9, 1.0), std::invalid_argument);
}

TEST(AdamTest, RejectsInvalidEpsilon) {
    Tensor parameter({1.0f}, true);

    EXPECT_THROW(Adam({parameter}, 0.01, 0.9, 0.999, 0.0), std::invalid_argument);

    EXPECT_THROW(Adam({parameter}, 0.01, 0.9, 0.999, -1e-8), std::invalid_argument);
}

TEST(AdamTest, RejectsNonFiniteHyperparameters) {
    Tensor parameter({1.0f}, true);

    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();

    EXPECT_THROW(Adam({parameter}, nan), std::invalid_argument);
    EXPECT_THROW(Adam({parameter}, inf), std::invalid_argument);

    EXPECT_THROW(Adam({parameter}, 0.01, nan), std::invalid_argument);
    EXPECT_THROW(Adam({parameter}, 0.01, 0.9, nan), std::invalid_argument);
    EXPECT_THROW(Adam({parameter}, 0.01, 0.9, 0.999, nan), std::invalid_argument);

    EXPECT_THROW(Adam({parameter}, 0.01, 0.9, 0.999, inf), std::invalid_argument);
}
