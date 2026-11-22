
#include "TestUtils.hpp"
#include "neuro/Tensor.hpp"

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <initializer_list>
#include <stdexcept>
#include <vector>

using namespace neuro;
using neuro::testing::expectArrayNear;

using Matrix = std::initializer_list<std::initializer_list<float>>;

// ==================== Forward Selection ====================

TEST(IndexSelect, ReorderedIndices) {
    Tensor x(Matrix{{10, 20}, {30, 40}, {50, 60}});

    Tensor y = x.indexSelect({2, 0});

    expectArrayNear(y.data(), af::dim4(2, 2), {50, 10, 60, 20});
}

TEST(IndexSelect, RepeatedIndices) {
    Tensor x(Matrix{{10, 20}, {30, 40}, {50, 60}});

    Tensor y = x.indexSelect({1, 1, 2});

    expectArrayNear(y.data(), af::dim4(3, 2), {30, 30, 50, 40, 40, 60});
}

TEST(IndexSelect, SingleIndex) {
    Tensor x(Matrix{{10, 20}, {30, 40}, {50, 60}});

    Tensor y = x.indexSelect({1});

    expectArrayNear(y.data(), af::dim4(1, 2), {30, 40});
}

TEST(IndexSelect, AllRowsReordered) {
    Tensor x({10, 20, 30, 40});

    Tensor y = x.indexSelect({3, 2, 1, 0});

    expectArrayNear(y.data(), af::dim4(4), {40, 30, 20, 10});
}

// ==================== Backward Propagation ====================

TEST(IndexSelectGrad, UniqueIndices) {
    Tensor x({10, 20, 30}, true);

    Tensor y = x.indexSelect({2, 0});
    y.sum().backward();

    expectArrayNear(x.grad(), af::dim4(3), {1, 0, 1});
}

TEST(IndexSelectGrad, RepeatedIndices) {
    Tensor x({10, 20, 30}, true);

    Tensor y = x.indexSelect({1, 1, 2});
    y.sum().backward();

    expectArrayNear(x.grad(), af::dim4(3), {0, 2, 1});
}

TEST(IndexSelectGrad, AllRepeatedIndices) {
    Tensor x({10, 20, 30}, true);

    Tensor y = x.indexSelect({0, 0, 0});
    y.sum().backward();

    expectArrayNear(x.grad(), af::dim4(3), {3, 0, 0});
}

TEST(IndexSelectGrad, NonUniformGradients) {
    Tensor x({2, 3, 4}, true);

    Tensor y = x.indexSelect({2, 0});
    Tensor weights({5, 7});

    (y * weights).sum().backward();

    // Output row 0 -> source row 2, weight 5
    // Output row 1 -> source row 0, weight 7
    expectArrayNear(x.grad(), af::dim4(3), {7, 0, 5});
}

TEST(IndexSelectGrad, RepeatedNonUniformGradients) {
    Tensor x({2, 3, 4}, true);

    Tensor y = x.indexSelect({1, 1, 2});
    Tensor weights({2, 5, 7});

    (y * weights).sum().backward();

    // Source row 1 receives 2 + 5
    // Source row 2 receives 7
    expectArrayNear(x.grad(), af::dim4(3), {0, 7, 7});
}

// ==================== Matrix Gradients ====================

TEST(IndexSelectGrad, MatrixSelection) {
    Tensor x(Matrix{{1, 2}, {3, 4}, {5, 6}}, true);

    Tensor y = x.indexSelect({2, 0});
    y.sum().backward();

    expectArrayNear(x.grad(), af::dim4(3, 2), {1, 0, 1, 1, 0, 1});
}

TEST(IndexSelectGrad, MatrixNonUniformGradients) {
    Tensor x(Matrix{{1, 2}, {3, 4}, {5, 6}}, true);

    Tensor weights(Matrix{{2, 3}, {5, 7}});

    Tensor y = x.indexSelect({2, 0});

    (y * weights).sum().backward();

    // Row 0 receives {5, 7}
    // Row 1 receives {0, 0}
    // Row 2 receives {2, 3}
    expectArrayNear(x.grad(), af::dim4(3, 2), {5, 0, 2, 7, 0, 3});
}

TEST(IndexSelectGrad, MatrixRepeatedIndices) {
    Tensor x(Matrix{{1, 2}, {3, 4}, {5, 6}}, true);

    Tensor y = x.indexSelect({1, 1, 2});
    y.sum().backward();

    expectArrayNear(x.grad(), af::dim4(3, 2), {0, 2, 1, 0, 2, 1});
}

// ==================== Higher Dimensions ====================

TEST(IndexSelect, ThreeDimensionalSelection) {
    af::array data = af::range(3, 2, 2);

    Tensor x(data, true);
    Tensor y = x.indexSelect({2, 0});

    EXPECT_EQ(y.shape()[0], 2);
    EXPECT_EQ(y.shape()[1], 2);
    EXPECT_EQ(y.shape()[2], 2);

    y.sum().backward();

    expectArrayNear(x.grad(), af::dim4(3, 2, 2), {1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1});
}

// ==================== Validation ====================

TEST(IndexSelectValidation, EmptyIndices) {
    Tensor x({10, 20, 30});

    EXPECT_THROW(x.indexSelect({}), std::invalid_argument);
}

TEST(IndexSelectValidation, NegativeIndex) {
    Tensor x({10, 20, 30});

    EXPECT_THROW(x.indexSelect({-1}), std::out_of_range);
}

TEST(IndexSelectValidation, OutOfBounds) {
    Tensor x({10, 20, 30});

    EXPECT_THROW(x.indexSelect({3}), std::out_of_range);
}

TEST(IndexSelectValidation, MixedValidAndInvalidIndices) {
    Tensor x({10, 20, 30});

    EXPECT_THROW(x.indexSelect({0, 2, 5}), std::out_of_range);
}

TEST(IndexSelectValidation, NoAutograd) {
    Tensor x({10, 20, 30}, false);

    Tensor y = x.indexSelect({2, 0});

    EXPECT_FALSE(y.requires_grad());

    expectArrayNear(y.data(), af::dim4(2), {30, 10});
}

TEST(IndexSelectValidation, PreservesGradTracking) {
    Tensor x({10, 20, 30}, true);

    Tensor y = x.indexSelect({2, 0});

    EXPECT_TRUE(y.requires_grad());
}
