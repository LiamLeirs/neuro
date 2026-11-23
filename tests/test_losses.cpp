#include "TestUtils.hpp"
#include "neuro/Tensor.hpp"
#include "neuro/nn/loss/CrossEntropyLoss.hpp"
#include "neuro/nn/loss/MAELoss.hpp"
#include "neuro/nn/loss/MSELoss.hpp"

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <initializer_list>
#include <stdexcept>

using namespace neuro;
using neuro::testing::expectArrayNear;
using Matrix = std::initializer_list<std::initializer_list<float>>;

TEST(MSELoss, Forward) {
    Tensor predicted({1, 3, 5});
    Tensor target({2, 5, 8});

    nn::MSELoss loss;

    EXPECT_NEAR(loss(predicted, target).data().scalar<float>(), 14.0f / 3.0f, 1e-5f);
}

TEST(MSELoss, Backward) {
    Tensor predicted({1, 2, 3}, true);
    Tensor target({2, 4, 6});

    nn::MSELoss loss;
    loss(predicted, target).backward();

    expectArrayNear(predicted.grad(), af::dim4(3), {-2.0f / 3.0f, -4.0f / 3.0f, -2.0f});
}

TEST(MSELoss, InvalidShape) {
    Tensor predicted({1, 2, 3});
    Tensor target({1, 2});

    nn::MSELoss loss;

    EXPECT_THROW(loss(predicted, target), std::invalid_argument);
}

TEST(MAELoss, Forward) {
    Tensor predicted({1, 3, 5});
    Tensor target({2, 5, 8});

    nn::MAELoss loss;

    EXPECT_NEAR(loss(predicted, target).data().scalar<float>(), 2.0f, 1e-5f);
}

TEST(MAELoss, Backward) {
    Tensor predicted({1, 3, 5, 7}, true);
    Tensor target({2, 2, 5, 4});

    nn::MAELoss loss;
    loss(predicted, target).backward();

    expectArrayNear(predicted.grad(), af::dim4(4), {-0.25f, 0.25f, 0.0f, 0.25f});
}

TEST(MAELoss, InvalidShape) {
    Tensor predicted({1, 2, 3});
    Tensor target({1, 2});

    nn::MAELoss loss;

    EXPECT_THROW(loss(predicted, target), std::invalid_argument);
}

TEST(CrossEntropyLoss, Forward) {
    Tensor logits(Matrix{{1.0f, 2.0f, 3.0f}, {2.0f, 1.0f, 0.0f}});

    nn::CrossEntropyLoss criterion;
    Tensor loss = criterion(logits, {2, 0});

    // Each sample has loss ≈ 0.407606
    EXPECT_NEAR(loss.data().scalar<float>(), 0.407606f, 1e-5f);
}

TEST(CrossEntropyLoss, Backward) {
    Tensor logits(Matrix{{1.0f, 2.0f, 3.0f}, {2.0f, 1.0f, 0.0f}}, true);

    nn::CrossEntropyLoss criterion;
    Tensor loss = criterion(logits, {2, 0});

    loss.backward();

    // Expected gradient:
    // (softmax(logits) - one_hot(target)) / batch_size
    //
    // Row 0: [ 0.045015, 0.122364, -0.167380]
    // Row 1: [-0.167380, 0.122364,  0.045015]
    //
    // Column-major ordering
    expectArrayNear(logits.grad(), af::dim4(2, 3),
                    {0.045015f, -0.167380f, 0.122364f, 0.122364f, -0.167380f, 0.045015f});
}

TEST(CrossEntropyLoss, NumericalStability) {
    Tensor logits(Matrix{{1000.0f, 1001.0f, 1002.0f}, {1002.0f, 1001.0f, 1000.0f}}, true);

    nn::CrossEntropyLoss criterion;
    Tensor loss = criterion(logits, {2, 0});

    float value = loss.data().scalar<float>();

    EXPECT_TRUE(std::isfinite(value));
    EXPECT_NEAR(value, 0.407606f, 1e-4f);

    loss.backward();

    af::array grad = logits.grad();
    std::vector<float> values(grad.elements());
    grad.host(values.data());

    for (float g : values) {
        EXPECT_TRUE(std::isfinite(g));
    }
}
