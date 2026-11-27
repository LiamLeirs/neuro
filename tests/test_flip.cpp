#include "neuro/Tensor.hpp"

#include <arrayfire.h>
#include <gtest/gtest.h>
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

} // namespace

TEST(FlipTest, ReversesVector) {
    Tensor input({1.0f, 2.0f, 3.0f, 4.0f});

    Tensor output = input.flip(0);

    EXPECT_EQ(output.shape(), input.shape());
    expectValues(output.data(), {4, 3, 2, 1});
}

TEST(FlipTest, ReversesMatrixDimensions) {
    const float values[]{1, 2, 3, 4, 5, 6};
    Tensor input(af::array(2, 3, values));

    expectValues(input.flip(0).data(), {2, 1, 4, 3, 6, 5});
    expectValues(input.flip(1).data(), {5, 6, 3, 4, 1, 2});
}

TEST(FlipTest, ReversesImageWidth) {
    const float values[]{1, 2, 3, 4, 5, 6};
    Tensor input(af::array(1, 2, 3, values));

    Tensor output = input.flip(2);

    EXPECT_EQ(output.shape(), input.shape());
    expectValues(output.data(), {5, 6, 3, 4, 1, 2});
}

TEST(FlipTest, ReversesBatchWidth) {
    const float values[]{1, 2, 3, 4, 5, 6, 7, 8};
    Tensor input(af::array(2, 1, 2, 2, values));

    Tensor output = input.flip(3);

    EXPECT_EQ(output.shape(), input.shape());
    expectValues(output.data(), {5, 6, 7, 8, 1, 2, 3, 4});
}

TEST(FlipTest, DoubleFlipRestoresInput) {
    Tensor input({1.0f, 2.0f, 3.0f, 4.0f});

    expectValues(input.flip(0).flip(0).data(), {1, 2, 3, 4});
}

TEST(FlipTest, SingletonDimensionIsUnchanged) {
    Tensor input({1.0f, 2.0f, 3.0f});

    expectValues(input.flip(1).data(), {1, 2, 3});
}

TEST(FlipTest, PreservesIntegerType) {
    const unsigned char values[]{1, 2, 3};
    Tensor input(af::array(3, values));

    Tensor output = input.flip(0);

    EXPECT_EQ(output.data().type(), u8);

    unsigned char actual[3];
    output.data().host(actual);

    EXPECT_EQ(actual[0], 3);
    EXPECT_EQ(actual[1], 2);
    EXPECT_EQ(actual[2], 1);
}

TEST(FlipTest, DoesNotModifyInput) {
    Tensor input({1.0f, 2.0f, 3.0f});

    Tensor output = input.flip(0);

    expectValues(input.data(), {1, 2, 3});
    expectValues(output.data(), {3, 2, 1});
}

TEST(FlipTest, BackwardReversesUpstreamGradient) {
    const float values[]{1, 2, 3, 4};
    Tensor input(af::array(4, values), true);

    Tensor weights({1.0f, 2.0f, 3.0f, 4.0f});

    Tensor loss = (input.flip(0) * weights).sum();
    loss.backward();

    expectValues(input.grad(), {4, 3, 2, 1});
}

TEST(FlipTest, BackwardAlongSecondDimension) {
    const float values[]{1, 2, 3, 4, 5, 6};
    const float weights[]{1, 2, 3, 4, 5, 6};

    Tensor input(af::array(2, 3, values), true);
    Tensor weight(af::array(2, 3, weights));

    Tensor loss = (input.flip(1) * weight).sum();
    loss.backward();

    expectValues(input.grad(), {5, 6, 3, 4, 1, 2});
}

TEST(FlipTest, RejectsInvalidDimension) {
    Tensor input({1.0f, 2.0f});

    EXPECT_THROW(input.flip(-1), std::invalid_argument);
    EXPECT_THROW(input.flip(4), std::invalid_argument);
}

TEST(FlipTest, RejectsEmptyInput) {
    Tensor input;

    EXPECT_THROW(input.flip(0), std::invalid_argument);
}