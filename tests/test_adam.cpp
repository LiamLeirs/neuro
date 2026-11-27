
#include "neuro/Tensor.hpp"
#include "neuro/optim/Adam.hpp"
#include "neuro/optim/AdamW.hpp"

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

TEST(AdamTest, RejectsInvalidWeightDecay) {
    Tensor param({2.0f}, true);

    EXPECT_THROW(Adam({param}, 0.1, 0.9, 0.999, 1e-8, -0.01), std::invalid_argument);

    EXPECT_THROW(Adam({param}, 0.1, 0.9, 0.999, 1e-8, std::numeric_limits<double>::infinity()),
                 std::invalid_argument);

    EXPECT_THROW(Adam({param}, 0.1, 0.9, 0.999, 1e-8, std::numeric_limits<double>::quiet_NaN()),
                 std::invalid_argument);
}

TEST(AdamTest, ZeroWeightDecayPreservesUpdate) {
    Tensor param1({2.0f}, true);
    Tensor param2({2.0f}, true);

    Adam optimizer1({param1}, 0.1);
    Adam optimizer2({param2}, 0.1, 0.9, 0.999, 1e-8, 0.0);

    for (int step = 0; step < 5; ++step) {
        optimizer1.zeroGrad();
        optimizer2.zeroGrad();

        (param1 * 3.0).sum().backward();
        (param2 * 3.0).sum().backward();

        optimizer1.step();
        optimizer2.step();

        EXPECT_NEAR(param1.data().scalar<float>(), 2.0 - 0.1 * (step + 1), 1e-5);

        EXPECT_FLOAT_EQ(param1.data().scalar<float>(), param2.data().scalar<float>());
    }
}

TEST(AdamTest, AppliesCoupledWeightDecayAcrossMultipleSteps) {
    constexpr double lr = 0.1;
    constexpr double b1 = 0.5;
    constexpr double b2 = 0.5;
    constexpr double eps = 1e-8;
    constexpr double decay = 0.2;

    Tensor param({2.0f}, true);
    Adam optimizer({param}, lr, b1, b2, eps, decay);

    double expected = 2.0;
    double m = 0.0;
    double v = 0.0;

    const double gradients[]{3.0, -1.0, 0.5, -2.0, 0.0};

    for (int t = 1; t <= 5; ++t) {
        optimizer.zeroGrad();

        const double g = gradients[t - 1];

        (param * g).sum().backward();

        optimizer.step();

        const double effective_grad = g + decay * expected;

        m = b1 * m + (1.0 - b1) * effective_grad;
        v = b2 * v + (1.0 - b2) * effective_grad * effective_grad;

        const double m_hat = m / (1.0 - std::pow(b1, t));
        const double v_hat = v / (1.0 - std::pow(b2, t));

        expected -= lr * m_hat / (std::sqrt(v_hat) + eps);

        EXPECT_NEAR(param.data().scalar<float>(), expected, 1e-5) << "Step: " << t;
    }
}

TEST(AdamTest, WeightDecayShrinksParametersWithZeroGradient) {
    Tensor param({2.0f, -4.0f}, true);
    Adam optimizer({param}, 0.1, 0.9, 0.999, 1e-8, 0.2);

    (param * 0.0).sum().backward();

    optimizer.step();

    float values[2];
    param.data().host(values);

    EXPECT_NEAR(values[0], 1.9, 1e-6);
    EXPECT_NEAR(values[1], -3.9, 1e-6);
}

TEST(AdamTest, WeightDecayDoesNotModifyGradients) {
    Tensor param({2.0f, -4.0f}, true);
    Adam optimizer({param}, 0.1, 0.9, 0.999, 1e-8, 0.2);

    (param * 3.0).sum().backward();

    optimizer.step();

    float gradients[2];
    param.grad().host(gradients);

    EXPECT_FLOAT_EQ(gradients[0], 3.0f);
    EXPECT_FLOAT_EQ(gradients[1], 3.0f);
}

TEST(AdamWTest, RejectsInvalidWeightDecay) {
    Tensor parameter({2.0f}, true);

    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();

    EXPECT_THROW(AdamW({parameter}, 0.1, 0.9, 0.999, 1e-8, -0.01), std::invalid_argument);

    EXPECT_THROW(AdamW({parameter}, 0.1, 0.9, 0.999, 1e-8, inf), std::invalid_argument);

    EXPECT_THROW(AdamW({parameter}, 0.1, 0.9, 0.999, 1e-8, nan), std::invalid_argument);
}

TEST(AdamWTest, UsesDefaultWeightDecay) {
    Tensor parameter({2.0f}, true);
    AdamW optimizer({parameter}, 0.1);

    backwardWithGradient(parameter, 0.0);
    optimizer.step();

    EXPECT_NEAR(scalarValue(parameter), 1.998, kTolerance);
}

TEST(AdamWTest, MatchesAdamWithoutWeightDecay) {
    Tensor first({2.0f}, true);
    Tensor second({2.0f}, true);

    Adam adam({first}, 0.1);
    AdamW adamw({second}, 0.1, 0.9, 0.999, 1e-8, 0.0);

    const double gradients[]{0.5, -0.25, 1.0, 0.0, -0.75};

    for (double gradient : gradients) {
        adam.zeroGrad();
        adamw.zeroGrad();

        backwardWithGradient(first, gradient);
        backwardWithGradient(second, gradient);

        adam.step();
        adamw.step();

        EXPECT_NEAR(scalarValue(first), scalarValue(second), kTolerance);
    }
}

TEST(AdamWTest, AppliesProportionalDecayWithZeroGradient) {
    Tensor parameter({2.0f, -4.0f}, true);
    AdamW optimizer({parameter}, 0.1, 0.9, 0.999, 1e-8, 0.2);

    backwardWithGradient(parameter, 0.0);
    optimizer.step();

    float values[2];
    parameter.data().host(values);

    EXPECT_NEAR(values[0], 1.96, kTolerance);
    EXPECT_NEAR(values[1], -3.92, kTolerance);
}

TEST(AdamWTest, AppliesDecoupledWeightDecayAcrossMultipleSteps) {
    constexpr double lr = 0.1;
    constexpr double decay = 0.2;

    Tensor parameter({2.0f}, true);
    AdamW optimizer({parameter}, lr, 0.9, 0.999, 1e-8, decay);

    ReferenceAdam reference{.value = 2.0, .lr = lr};

    const double gradients[]{3.0, -1.0, 0.5, -2.0, 0.0};

    for (double gradient : gradients) {
        optimizer.zeroGrad();

        backwardWithGradient(parameter, gradient);
        optimizer.step();

        reference.value *= 1.0 - lr * decay;
        reference.step(gradient);

        EXPECT_NEAR(scalarValue(parameter), reference.value, kTolerance);
    }
}

TEST(AdamWTest, WeightDecayDoesNotInfluenceMoments) {
    constexpr double lr = 0.1;
    constexpr double decay = 0.2;

    Tensor first({2.0f}, true);
    Tensor second({2.0f}, true);

    Adam adam({first}, lr);
    AdamW adamw({second}, lr, 0.9, 0.999, 1e-8, decay);

    const double gradients[]{0.5, -0.25, 0.0, 1.0, -0.75};

    for (double gradient : gradients) {
        adam.zeroGrad();
        adamw.zeroGrad();

        backwardWithGradient(first, gradient);
        backwardWithGradient(second, gradient);

        const double beforeAdam = scalarValue(first);
        const double beforeAdamW = scalarValue(second);

        adam.step();
        adamw.step();

        const double adamUpdate = beforeAdam - scalarValue(first);

        const double adamwAdaptiveUpdate = beforeAdamW * (1.0 - lr * decay) - scalarValue(second);

        EXPECT_NEAR(adamUpdate, adamwAdaptiveUpdate, kTolerance);
    }
}

TEST(AdamWTest, WeightDecayDoesNotModifyGradients) {
    Tensor parameter({2.0f, -4.0f}, true);
    AdamW optimizer({parameter}, 0.1, 0.9, 0.999, 1e-8, 0.2);

    backwardWithGradient(parameter, 3.0);
    optimizer.step();

    float gradients[2];
    parameter.grad().host(gradients);

    EXPECT_FLOAT_EQ(gradients[0], 3.0f);
    EXPECT_FLOAT_EQ(gradients[1], 3.0f);
}

TEST(AdamWTest, IgnoresParametersWithoutGradientTracking) {
    Tensor frozen({1.0f}, false);
    Tensor trainable({2.0f}, true);

    AdamW optimizer({frozen, trainable}, 0.1, 0.9, 0.999, 1e-8, 0.2);

    backwardWithGradient(trainable, 0.5);
    optimizer.step();

    EXPECT_FLOAT_EQ(scalarValue(frozen), 1.0f);
    EXPECT_NEAR(scalarValue(trainable), 1.86, kTolerance);
}