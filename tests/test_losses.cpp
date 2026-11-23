#include "TestUtils.hpp"
#include "neuro/Tensor.hpp"
#include "neuro/nn/MAELoss.hpp"
#include "neuro/nn/MSELoss.hpp"

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <initializer_list>
#include <stdexcept>

using namespace neuro;
using neuro::testing::expectArrayNear;

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
