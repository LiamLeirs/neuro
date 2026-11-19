
#include "TestUtils.hpp"
#include "neuro/Tensor.hpp"
#include "neuro/optim/SGD.hpp"

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <initializer_list>
#include <vector>

using namespace neuro;
using neuro::testing::expectArrayNear;

using Matrix = std::initializer_list<std::initializer_list<float>>;

// ==================== Basic SGD ====================

TEST(SGD, SingleParameterUpdate) {
    Tensor w({2.0f}, true);
    SGD optimizer({w}, 0.1f);

    // L = w², dL/dw = 2w = 4
    (w * w).backward();

    optimizer.step();

    // w_new = 2 - 0.1 * 4 = 1.6
    expectArrayNear(w.data(), af::dim4(1), {1.6f});
}

TEST(SGD, MultipleParameterUpdate) {
    Tensor w({2.0f}, true);
    Tensor b({3.0f}, true);

    SGD optimizer({w, b}, 0.1f);

    // L = w² + b²
    Tensor loss = w * w + b * b;

    loss.backward();
    optimizer.step();

    expectArrayNear(w.data(), af::dim4(1), {1.6f});
    expectArrayNear(b.data(), af::dim4(1), {2.4f});
}

TEST(SGD, VectorParameterUpdate) {
    Tensor w({1.0f, 2.0f, 3.0f}, true);
    SGD optimizer({w}, 0.1f);

    // L = sum(w²)
    (w * w).sum().backward();

    expectArrayNear(w.grad(), af::dim4(3), {2, 4, 6});

    optimizer.step();

    expectArrayNear(w.data(), af::dim4(3), {0.8f, 1.6f, 2.4f});
}

TEST(SGD, MatrixParameterUpdate) {
    Tensor w(Matrix{{1, 2}, {3, 4}}, true);

    SGD optimizer({w}, 0.1f);

    (w * w).sum().backward();
    optimizer.step();

    expectArrayNear(w.data(), af::dim4(2, 2), {0.8f, 2.4f, 1.6f, 3.2f});
}

// ==================== Gradient Management ====================

TEST(SGD, ZeroGrad) {
    Tensor w({2.0f}, true);
    Tensor b({3.0f}, true);

    SGD optimizer({w, b}, 0.1f);

    (w * w + b * b).backward();

    expectArrayNear(w.grad(), af::dim4(1), {4});
    expectArrayNear(b.grad(), af::dim4(1), {6});

    optimizer.zeroGrad();

    expectArrayNear(w.grad(), af::dim4(1), {0});
    expectArrayNear(b.grad(), af::dim4(1), {0});
}

TEST(SGD, GradientsUnaffectedByStep) {
    Tensor w({2.0f}, true);
    SGD optimizer({w}, 0.1f);

    (w * w).backward();
    optimizer.step();

    // step() updates parameters but should not zero gradients.
    expectArrayNear(w.grad(), af::dim4(1), {4});
}

TEST(SGD, MultipleSteps) {
    Tensor w({2.0f}, true);
    SGD optimizer({w}, 0.1f);

    for (int i = 0; i < 2; ++i) {
        optimizer.zeroGrad();

        Tensor loss = w * w;
        loss.backward();

        optimizer.step();
    }

    // Step 1: 2.0 -> 1.6
    // Step 2: 1.6 -> 1.28
    expectArrayNear(w.data(), af::dim4(1), {1.28f});
}

// ==================== Training Integration ====================

TEST(SGDIntegration, LinearRegression) {
    Tensor w({0.0f}, true);
    Tensor b({0.0f}, true);

    Tensor x({-2, -1, 0, 1, 2});
    Tensor target({-3, -1, 1, 3, 5});

    SGD optimizer({w, b}, 0.01f);

    for (int epoch = 0; epoch < 500; ++epoch) {
        Tensor prediction = x.matmul(w) + b;
        Tensor error = prediction - target;

        Tensor loss = (error * error).sum() / 5.0;

        optimizer.zeroGrad();
        loss.backward();
        optimizer.step();
    }

    EXPECT_NEAR(w.data().scalar<float>(), 2.0f, 1e-3f);
    EXPECT_NEAR(b.data().scalar<float>(), 1.0f, 1e-3f);
}

TEST(SGDIntegration, MultiFeatureRegression) {
    Tensor w(Matrix{{0.0f}, {0.0f}}, true);

    Tensor b({0.0f}, true);

    Tensor x(Matrix{{-2, -1}, {-1, 2}, {0, -2}, {1, 1}, {2, 0}, {3, -1}, {-3, 2}, {2, 3}});

    Tensor target(Matrix{{0}, {-7}, {7}, {0}, {5}, {10}, {-11}, {-4}});

    SGD optimizer({w, b}, 0.01f);

    for (int epoch = 0; epoch < 500; ++epoch) {
        Tensor prediction = x.matmul(w) + b;
        Tensor error = prediction - target;
        Tensor loss = (error * error).sum() / 8.0;

        optimizer.zeroGrad();
        loss.backward();
        optimizer.step();
    }

    EXPECT_NEAR(w.data()(0).scalar<float>(), 2.0f, 1e-3f);
    EXPECT_NEAR(w.data()(1).scalar<float>(), -3.0f, 1e-3f);
    EXPECT_NEAR(b.data().scalar<float>(), 1.0f, 1e-3f);
}
