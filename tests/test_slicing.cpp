
#include "TestUtils.hpp"
#include "neuro/Tensor.hpp"

#include <arrayfire.h>
#include <gtest/gtest.h>

using namespace neuro;
using neuro::testing::expectArrayNear;

TEST(TensorSlicing, Forward) {
    Tensor x(af::iota(af::dim4(4, 3), af::dim4(1), f32), true);

    Tensor y = x.slice(1, 3);

    expectArrayNear(y.data(), af::dim4(2, 3), {1, 2, 5, 6, 9, 10});
}

TEST(TensorSlicing, Backward) {
    Tensor x(af::iota(af::dim4(4, 3), af::dim4(1), f32), true);

    Tensor y = x.slice(1, 3);
    Tensor loss = (y * y).sum();

    loss.backward();

    expectArrayNear(x.grad(), af::dim4(4, 3), {0, 2, 4, 0, 0, 10, 12, 0, 0, 18, 20, 0});
}

TEST(TensorSlicing, OverlappingGradients) {
    Tensor x(af::constant(1.0f, 5, 2), true);

    Tensor a = x.slice(1, 4);
    Tensor b = x.slice(2, 5);

    Tensor loss = a.sum() + b.sum();
    loss.backward();

    expectArrayNear(x.grad(), af::dim4(5, 2), {0, 1, 2, 2, 1, 0, 1, 2, 2, 1});
}

TEST(TensorSlicing, InvalidRanges) {
    Tensor x(af::constant(1.0f, 4, 3), true);

    EXPECT_THROW(x.slice(2, 2), std::invalid_argument);
    EXPECT_THROW(x.slice(3, 2), std::invalid_argument);
    EXPECT_THROW(x.slice(-1, 2), std::out_of_range);
    EXPECT_THROW(x.slice(0, 5), std::out_of_range);
}

TEST(TensorSlicing, NoAutograd) {
    Tensor x(af::constant(1.0f, 4, 3), false);

    Tensor y = x.slice(1, 3);

    EXPECT_FALSE(y.requires_grad());
    EXPECT_EQ(y.shape(), af::dim4(2, 3));
}
