
#include "TestUtils.hpp"
#include "neuro/Tensor.hpp"

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <initializer_list>

using namespace neuro;
using neuro::testing::expectArrayNear;

using Matrix = std::initializer_list<std::initializer_list<float>>;

// ==================== Forward Propagation ====================

TEST(ReLU, MixedValues) {
    Tensor x({-3, -1, 0, 2, 5});

    Tensor y = x.relu();

    expectArrayNear(y.data(), af::dim4(5), {0, 0, 0, 2, 5});
}

TEST(ReLU, AllNegative) {
    Tensor x({-5, -3, -1});

    Tensor y = x.relu();

    expectArrayNear(y.data(), af::dim4(3), {0, 0, 0});
}

TEST(ReLU, AllPositive) {
    Tensor x({1, 2, 3});

    Tensor y = x.relu();

    expectArrayNear(y.data(), af::dim4(3), {1, 2, 3});
}

TEST(ReLU, MatrixForward) {
    Tensor x(Matrix{{-2, 0, 3}, {4, -1, 5}});

    Tensor y = x.relu();

    expectArrayNear(y.data(), af::dim4(2, 3), {0, 4, 0, 0, 3, 5});
}

TEST(ReLU, PreservesShape) {
    Tensor x(af::randu(4, 3, 2));

    Tensor y = x.relu();

    EXPECT_EQ(y.shape(), af::dim4(4, 3, 2));
}

// ==================== Backward Propagation ====================

TEST(ReLUGrad, MixedValues) {
    Tensor x({-3, -1, 0, 2, 5}, true);

    x.relu().sum().backward();

    expectArrayNear(x.grad(), af::dim4(5), {0, 0, 0, 1, 1});
}

TEST(ReLUGrad, AllNegative) {
    Tensor x({-5, -3, -1}, true);

    x.relu().sum().backward();

    expectArrayNear(x.grad(), af::dim4(3), {0, 0, 0});
}

TEST(ReLUGrad, AllPositive) {
    Tensor x({1, 2, 3}, true);

    x.relu().sum().backward();

    expectArrayNear(x.grad(), af::dim4(3), {1, 1, 1});
}

TEST(ReLUGrad, ZeroDerivative) {
    Tensor x({0}, true);

    x.relu().sum().backward();

    expectArrayNear(x.grad(), af::dim4(1), {0});
}

TEST(ReLUGrad, NonUniformUpstreamGradients) {
    Tensor x({-2, 0, 3, 5}, true);
    Tensor weights({10, 20, 30, 40});

    Tensor y = x.relu();
    Tensor loss = (y * weights).sum();

    loss.backward();

    expectArrayNear(x.grad(), af::dim4(4), {0, 0, 30, 40});
}

TEST(ReLUGrad, MatrixBackward) {
    Tensor x(Matrix{{-2, 0, 3}, {4, -1, 5}}, true);

    x.relu().sum().backward();

    expectArrayNear(x.grad(), af::dim4(2, 3), {0, 1, 0, 0, 1, 1});
}

TEST(ReLUGrad, ThreeDimensionalBackward) {
    af::array data = af::constant(-2.0f, 2, 2, 2);

    // Set one element positive.
    data(1, 0, 1) = 3.0f;

    Tensor x(data, true);

    x.relu().sum().backward();

    expectArrayNear(x.grad(), af::dim4(2, 2, 2), {0, 0, 0, 0, 0, 1, 0, 0});
}

// ==================== Computation Graph ====================

TEST(ReLUGrad, RepeatedUseAccumulatesGradients) {
    Tensor x({-2, 3, 5}, true);

    Tensor a = x.relu();
    Tensor b = x.relu();

    Tensor loss = (a + b).sum();
    loss.backward();

    expectArrayNear(x.grad(), af::dim4(3), {0, 2, 2});
}

TEST(ReLUGrad, ChainedReLU) {
    Tensor x({-2, 0, 3}, true);

    Tensor y = x.relu().relu();
    y.sum().backward();

    expectArrayNear(x.grad(), af::dim4(3), {0, 0, 1});
}

TEST(ReLUGrad, ComposedWithArithmetic) {
    Tensor x({-2, 1, 3}, true);

    Tensor y = (x * 2.0f).relu();
    y.sum().backward();

    // d/dx ReLU(2x) = 2 when x > 0
    expectArrayNear(x.grad(), af::dim4(3), {0, 2, 2});
}

// ==================== Gradient Tracking ====================

TEST(ReLU, NoGradientTracking) {
    Tensor x({-2, 0, 3}, false);

    Tensor y = x.relu();

    EXPECT_FALSE(y.requires_grad());

    expectArrayNear(y.data(), af::dim4(3), {0, 0, 3});
}

TEST(ReLU, PreservesGradientTracking) {
    Tensor x({-2, 0, 3}, true);

    Tensor y = x.relu();

    EXPECT_TRUE(y.requires_grad());
}

TEST(AutogradSigmoid, Forward) {
    Tensor x({-1.0f, 0.0f, 1.0f});

    Tensor result = x.sigmoid();

    expectArrayNear(result.data(), af::dim4(3), {0.268941f, 0.5f, 0.731059f});
}

TEST(AutogradSigmoid, Backward) {
    Tensor x({-1.0f, 0.0f, 1.0f}, true);

    x.sigmoid().sum().backward();

    expectArrayNear(x.grad(), af::dim4(3), {0.196612f, 0.25f, 0.196612f});
}

TEST(AutogradSigmoid, NumericalStability) {
    Tensor x({-1000.0f, 1000.0f}, true);

    Tensor result = x.sigmoid();

    expectArrayNear(result.data(), af::dim4(2), {0.0f, 1.0f});

    result.sum().backward();

    expectArrayNear(x.grad(), af::dim4(2), {0.0f, 0.0f});
}

TEST(AutogradTanh, Forward) {
    Tensor x({-1.0f, 0.0f, 1.0f});

    Tensor result = x.tanh();

    expectArrayNear(result.data(), af::dim4(3), {-0.761594f, 0.0f, 0.761594f});
}

TEST(AutogradTanh, Backward) {
    Tensor x({-1.0f, 0.0f, 1.0f}, true);

    x.tanh().sum().backward();

    expectArrayNear(x.grad(), af::dim4(3), {0.419974f, 1.0f, 0.419974f});
}

TEST(AutogradTanh, NumericalStability) {
    Tensor x({-1000.0f, 1000.0f}, true);

    Tensor result = x.tanh();

    expectArrayNear(result.data(), af::dim4(2), {-1.0f, 1.0f});

    result.sum().backward();

    expectArrayNear(x.grad(), af::dim4(2), {0.0f, 0.0f});
}
