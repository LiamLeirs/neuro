#include "neuro/Tensor.hpp"

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace neuro;

namespace {

void expectValues(const af::array& array, const std::vector<float>& expected) {
    ASSERT_EQ(array.elements(), expected.size());

    std::vector<float> actual(expected.size());
    array.host(actual.data());

    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(actual[i], expected[i]) << "Index: " << i;
    }
}

Tensor makeMask(const std::vector<unsigned char>& values) {
    return Tensor(af::array(values.size(), values.data()).as(b8));
}

} // namespace

TEST(WhereTest, SelectsElementsFromBothInputs) {
    Tensor condition = makeMask({1, 0, 1, 0});
    Tensor x({1.0f, 2.0f, 3.0f, 4.0f});
    Tensor y({5.0f, 6.0f, 7.0f, 8.0f});

    Tensor output = Tensor::where(condition, x, y);

    EXPECT_EQ(output.shape(), x.shape());
    expectValues(output.data(), {1, 6, 3, 8});
}

TEST(WhereTest, HandlesUniformConditions) {
    Tensor x({1.0f, 2.0f, 3.0f});
    Tensor y({4.0f, 5.0f, 6.0f});

    expectValues(Tensor::where(makeMask({1, 1, 1}), x, y).data(), {1, 2, 3});

    expectValues(Tensor::where(makeMask({0, 0, 0}), x, y).data(), {4, 5, 6});
}

TEST(WhereTest, BroadcastsCondition) {
    Tensor condition = makeMask({1, 0});

    const float x_values[]{1, 2, 3, 4, 5, 6};
    const float y_values[]{7, 8, 9, 10, 11, 12};

    Tensor x(af::array(2, 3, x_values));
    Tensor y(af::array(2, 3, y_values));

    Tensor output = Tensor::where(condition, x, y);

    EXPECT_EQ(output.shape(), af::dim4(2, 3));
    expectValues(output.data(), {1, 8, 3, 10, 5, 12});
}

TEST(WhereTest, BroadcastsScalarOperand) {
    Tensor condition = makeMask({1, 0, 1, 0});
    Tensor x({10.0f});
    Tensor y({1.0f, 2.0f, 3.0f, 4.0f});

    expectValues(Tensor::where(condition, x, y).data(), {10, 2, 10, 4});

    expectValues(Tensor::where(condition, y, x).data(), {1, 10, 3, 10});
}

TEST(WhereTest, BroadcastsBothValueOperands) {
    const unsigned char mask_values[]{1, 0, 0, 1, 1, 0};
    const float x_values[]{10, 20};
    const float y_values[]{1, 2, 3};

    Tensor condition(af::array(2, 3, mask_values).as(b8));
    Tensor x(af::array(2, 1, x_values));
    Tensor y(af::array(1, 3, y_values));

    Tensor output = Tensor::where(condition, x, y);

    EXPECT_EQ(output.shape(), af::dim4(2, 3));
    expectValues(output.data(), {10, 1, 2, 20, 10, 3});
}

TEST(WhereTest, BroadcastsAllThreeOperands) {
    Tensor condition = makeMask({1, 0});

    const float x_values[]{10, 20, 30};

    Tensor x(af::array(1, 3, x_values));
    Tensor y({5.0f});

    Tensor output = Tensor::where(condition, x, y);

    EXPECT_EQ(output.shape(), af::dim4(2, 3));
    expectValues(output.data(), {10, 5, 20, 5, 30, 5});
}

TEST(WhereTest, BroadcastsImageBatchMask) {
    Tensor condition = makeMask({1, 0});

    Tensor x(af::constant(10.0f, 2, 3, 2, 4));
    Tensor y(af::constant(20.0f, 2, 3, 2, 4));

    Tensor output = Tensor::where(condition, x, y);

    EXPECT_EQ(output.shape(), af::dim4(2, 3, 2, 4));

    std::vector<float> expected(48);

    for (std::size_t i = 0; i < expected.size(); ++i) {
        expected[i] = i % 2 == 0 ? 10.0f : 20.0f;
    }

    expectValues(output.data(), expected);
}

TEST(WhereTest, PreservesIntegerType) {
    Tensor condition = makeMask({1, 0, 1});

    const unsigned char x_values[]{1, 2, 3};
    const unsigned char y_values[]{4, 5, 6};

    Tensor x(af::array(3, x_values));
    Tensor y(af::array(3, y_values));

    Tensor output = Tensor::where(condition, x, y);

    EXPECT_EQ(output.data().type(), u8);

    unsigned char actual[3];
    output.data().host(actual);

    EXPECT_EQ(actual[0], 1);
    EXPECT_EQ(actual[1], 5);
    EXPECT_EQ(actual[2], 3);
}

TEST(WhereTest, IgnoresUnselectedNonFiniteValues) {
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();

    Tensor condition = makeMask({1, 0});

    Tensor x({1.0f, nan});
    Tensor y({inf, 2.0f});

    expectValues(Tensor::where(condition, x, y).data(), {1, 2});
}

TEST(WhereTest, BackwardRoutesGradients) {
    Tensor condition = makeMask({1, 0, 1, 0});

    const float x_values[]{1, 2, 3, 4};
    const float y_values[]{5, 6, 7, 8};

    Tensor x(af::array(4, x_values), true);
    Tensor y(af::array(4, y_values), true);

    Tensor weights({2.0f, 3.0f, 4.0f, 5.0f});

    Tensor loss = (Tensor::where(condition, x, y) * weights).sum();
    loss.backward();

    expectValues(x.grad(), {2, 0, 4, 0});
    expectValues(y.grad(), {0, 3, 0, 5});
}

TEST(WhereTest, BackwardSupportsSingleTrainableOperand) {
    Tensor condition = makeMask({1, 0, 1});

    const float values[]{1, 2, 3};

    Tensor x(af::array(3, values), true);
    Tensor y({4.0f, 5.0f, 6.0f});

    Tensor::where(condition, x, y).sum().backward();

    expectValues(x.grad(), {1, 0, 1});
    EXPECT_FALSE(y.requires_grad());
}

TEST(WhereTest, BackwardAccumulatesSharedParent) {
    Tensor condition = makeMask({1, 0, 1, 0});

    const float values[]{1, 2, 3, 4};
    Tensor input(af::array(4, values), true);

    Tensor weights({2.0f, 3.0f, 4.0f, 5.0f});

    Tensor loss = (Tensor::where(condition, input, input) * weights).sum();

    loss.backward();

    expectValues(input.grad(), {2, 3, 4, 5});
}

TEST(WhereTest, BackwardReducesBroadcastedScalar) {
    Tensor condition = makeMask({1, 0, 1, 1});

    const float x_value[]{5};
    const float y_values[]{1, 2, 3, 4};

    Tensor x(af::array(1, x_value), true);
    Tensor y(af::array(4, y_values), true);

    Tensor weights({2.0f, 3.0f, 4.0f, 5.0f});

    Tensor loss = (Tensor::where(condition, x, y) * weights).sum();

    loss.backward();

    EXPECT_EQ(x.grad().dims(), x.shape());
    EXPECT_EQ(y.grad().dims(), y.shape());

    expectValues(x.grad(), {11});
    expectValues(y.grad(), {0, 3, 0, 0});
}

TEST(WhereTest, BackwardReducesBothBroadcastedOperands) {
    const unsigned char mask_values[]{1, 0, 0, 1, 1, 0};
    const float x_values[]{10, 20};
    const float y_values[]{1, 2, 3};
    const float weights[]{1, 2, 3, 4, 5, 6};

    Tensor condition(af::array(2, 3, mask_values).as(b8));
    Tensor x(af::array(2, 1, x_values), true);
    Tensor y(af::array(1, 3, y_values), true);
    Tensor weight(af::array(2, 3, weights));

    Tensor loss = (Tensor::where(condition, x, y) * weight).sum();

    loss.backward();

    EXPECT_EQ(x.grad().dims(), x.shape());
    EXPECT_EQ(y.grad().dims(), y.shape());

    expectValues(x.grad(), {6, 4});
    expectValues(y.grad(), {2, 3, 6});
}

TEST(WhereTest, BackwardBroadcastsCondition) {
    Tensor condition = makeMask({1, 0});

    const float x_values[]{1, 2, 3, 4, 5, 6};
    const float y_values[]{7, 8, 9, 10, 11, 12};
    const float weights[]{1, 2, 3, 4, 5, 6};

    Tensor x(af::array(2, 3, x_values), true);
    Tensor y(af::array(2, 3, y_values), true);
    Tensor weight(af::array(2, 3, weights));

    Tensor loss = (Tensor::where(condition, x, y) * weight).sum();

    loss.backward();

    expectValues(x.grad(), {1, 0, 3, 0, 5, 0});
    expectValues(y.grad(), {0, 2, 0, 4, 0, 6});
}

TEST(WhereTest, RejectsEmptyOperands) {
    Tensor empty;
    Tensor condition = makeMask({1});
    Tensor input({1.0f});

    EXPECT_THROW(Tensor::where(empty, input, input), std::invalid_argument);

    EXPECT_THROW(Tensor::where(condition, empty, input), std::invalid_argument);

    EXPECT_THROW(Tensor::where(condition, input, empty), std::invalid_argument);
}

TEST(WhereTest, RejectsNonBooleanCondition) {
    Tensor condition({1.0f, 0.0f});
    Tensor input({1.0f, 2.0f});

    EXPECT_THROW(Tensor::where(condition, input, input), std::invalid_argument);
}

TEST(WhereTest, RejectsMismatchedDataTypes) {
    Tensor condition = makeMask({1, 0});

    Tensor x({1.0f, 2.0f});
    Tensor y(af::constant(1.0, 2, f64));

    EXPECT_THROW(Tensor::where(condition, x, y), std::invalid_argument);
}

TEST(WhereTest, RejectsIncompatibleShapes) {
    Tensor condition = makeMask({1, 0});
    Tensor x({1.0f, 2.0f, 3.0f});
    Tensor y({4.0f, 5.0f, 6.0f});

    EXPECT_THROW(Tensor::where(condition, x, y), af::exception);
}