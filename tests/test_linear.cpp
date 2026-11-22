
#include "TestUtils.hpp"
#include "neuro/Tensor.hpp"
#include "neuro/nn/Linear.hpp"

#include <arrayfire.h>
#include <cmath>
#include <gtest/gtest.h>
#include <initializer_list>
#include <stdexcept>
#include <vector>

using namespace neuro;
using namespace neuro::nn;
using neuro::testing::expectArrayNear;

using Matrix = std::initializer_list<std::initializer_list<float>>;

// ==================== Parameter Initialization ====================

TEST(Linear, ParameterShapes) {
    Linear layer(10, 5);

    EXPECT_EQ(layer.weight().shape(), af::dim4(10, 5));
    EXPECT_EQ(layer.bias().shape(), af::dim4(1, 5));
}

TEST(Linear, ParametersRequireGrad) {
    Linear layer(10, 5);

    EXPECT_TRUE(layer.weight().requires_grad());
    EXPECT_TRUE(layer.bias().requires_grad());
}

TEST(Linear, XavierInitializationBounds) {
    constexpr int inFeatures = 100;
    constexpr int outFeatures = 50;

    Linear layer(inFeatures, outFeatures);

    const double limit = std::sqrt(6.0 / (inFeatures + outFeatures));

    std::vector<float> weights(layer.weight().data().elements());
    layer.weight().data().host(weights.data());

    for (float weight : weights) {
        EXPECT_GE(weight, -limit);
        EXPECT_LE(weight, limit);
    }
}

TEST(Linear, BiasInitializedToZero) {
    Linear layer(3, 4);

    expectArrayNear(layer.bias().data(), af::dim4(1, 4), {0, 0, 0, 0});
}

// ==================== Forward Propagation ====================

TEST(Linear, ForwardShape) {
    Linear layer(10, 5);

    Tensor input(af::randu(32, 10));

    Tensor output = layer.forward(input);

    EXPECT_EQ(output.shape(), af::dim4(32, 5));
}

TEST(Linear, ForwardKnownValues) {
    Linear layer(2, 2);

    layer.weight() = Tensor(Matrix{{2, 1}, {1, 3}}, true);

    layer.bias() = Tensor(Matrix{{1, 2}}, true);

    Tensor input(Matrix{{1, 2}, {3, 4}});

    Tensor output = layer.forward(input);

    expectArrayNear(output.data(), af::dim4(2, 2), {5, 11, 9, 17});
}

TEST(Linear, CallOperatorMatchesForward) {
    Linear layer(2, 3);

    Tensor input(Matrix{{1, 2}, {3, 4}});

    Tensor a = layer.forward(input);
    Tensor b = layer(input);

    std::vector<float> expected(a.data().elements());
    a.data().host(expected.data());

    expectArrayNear(b.data(), af::dim4(2, 3), expected);
}

// ==================== Backward Propagation ====================

TEST(LinearGrad, WeightGradients) {
    Linear layer(2, 2);

    layer.weight() = Tensor(Matrix{{2, 1}, {1, 3}}, true);

    layer.bias() = Tensor(Matrix{{1, 2}}, true);

    Tensor input(Matrix{{1, 2}, {3, 4}});

    Tensor output = layer(input);
    output.sum().backward();

    // dL/dW = X^T * ones
    expectArrayNear(layer.weight().grad(), af::dim4(2, 2), {4, 6, 4, 6});
}

TEST(LinearGrad, BiasGradients) {
    Linear layer(2, 2);

    layer.weight() = Tensor(Matrix{{2, 1}, {1, 3}}, true);

    layer.bias() = Tensor(Matrix{{1, 2}}, true);

    Tensor input(Matrix{{1, 2}, {3, 4}});

    layer(input).sum().backward();

    // Each output neuron contributes once per sample.
    expectArrayNear(layer.bias().grad(), af::dim4(1, 2), {2, 2});
}

TEST(LinearGrad, InputGradients) {
    Linear layer(2, 2);

    layer.weight() = Tensor(Matrix{{2, 1}, {1, 3}}, true);

    layer.bias() = Tensor(Matrix{{1, 2}}, true);

    Tensor input(Matrix{{1, 2}, {3, 4}}, true);

    layer(input).sum().backward();

    // dL/dX = ones * W^T
    expectArrayNear(input.grad(), af::dim4(2, 2), {3, 3, 4, 4});
}

TEST(LinearGrad, AllGradientsTogether) {
    Linear layer(2, 2);

    layer.weight() = Tensor(Matrix{{2, 1}, {1, 3}}, true);

    layer.bias() = Tensor(Matrix{{1, 2}}, true);

    Tensor input(Matrix{{1, 2}, {3, 4}}, true);

    Tensor output = layer(input);
    output.sum().backward();

    expectArrayNear(layer.weight().grad(), af::dim4(2, 2), {4, 6, 4, 6});

    expectArrayNear(layer.bias().grad(), af::dim4(1, 2), {2, 2});

    expectArrayNear(input.grad(), af::dim4(2, 2), {3, 3, 4, 4});
}

TEST(LinearGrad, DifferentBatchSizes) {
    Linear layer(2, 3);

    layer.weight() = Tensor(Matrix{{1, 2, 3}, {4, 5, 6}}, true);

    layer.bias() = Tensor(Matrix{{0, 0, 0}}, true);

    Tensor input(Matrix{{1, 2}, {3, 4}, {5, 6}, {7, 8}});

    layer(input).sum().backward();

    // Feature sums: 1+3+5+7=16, 2+4+6+8=20
    expectArrayNear(layer.weight().grad(), af::dim4(2, 3), {16, 20, 16, 20, 16, 20});

    expectArrayNear(layer.bias().grad(), af::dim4(1, 3), {4, 4, 4});
}

// ==================== Parameter Reuse ====================

TEST(Linear, RepeatedForwardUsesSameParameters) {
    Linear layer(2, 1);

    layer.weight() = Tensor(Matrix{{2}, {3}}, true);

    layer.bias() = Tensor(Matrix{{1}}, true);

    Tensor input(Matrix{{1, 2}, {3, 4}});

    Tensor first = layer(input);
    Tensor second = layer(input);

    expectArrayNear(first.data(), af::dim4(2, 1), {9, 19});

    expectArrayNear(second.data(), af::dim4(2, 1), {9, 19});
}

TEST(LinearGrad, SharedParametersAccumulateGradients) {
    Linear layer(2, 1);

    layer.weight() = Tensor(Matrix{{2}, {3}}, true);

    layer.bias() = Tensor(Matrix{{1}}, true);

    Tensor firstInput(Matrix{{1, 2}});
    Tensor secondInput(Matrix{{3, 4}});

    Tensor firstOutput = layer(firstInput);
    Tensor secondOutput = layer(secondInput);

    Tensor loss = firstOutput.sum() + secondOutput.sum();
    loss.backward();

    expectArrayNear(layer.weight().grad(), af::dim4(2, 1), {4, 6});

    expectArrayNear(layer.bias().grad(), af::dim4(1, 1), {2});
}

// ==================== Validation ====================

TEST(LinearValidation, InvalidInputFeatures) {
    Linear layer(3, 2);

    Tensor input(af::randu(4, 5));

    EXPECT_THROW(layer(input), std::invalid_argument);
}

TEST(LinearValidation, InvalidInputRank) {
    Linear layer(3, 2);

    Tensor input(af::randu(4, 3, 2));

    EXPECT_THROW(layer(input), std::invalid_argument);
}

TEST(LinearValidation, InvalidConstructorDimensions) {
    EXPECT_THROW(Linear(0, 5), std::invalid_argument);
    EXPECT_THROW(Linear(5, 0), std::invalid_argument);
    EXPECT_THROW(Linear(-1, 5), std::invalid_argument);
    EXPECT_THROW(Linear(5, -1), std::invalid_argument);
}
