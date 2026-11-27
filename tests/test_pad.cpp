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

TEST(PadTest, SymmetricVectorPadding) {
    Tensor input({1.0f, 2.0f, 3.0f});

    Tensor output = input.pad({2, 0, 0, 0}, {2, 0, 0, 0});

    EXPECT_EQ(output.shape(), af::dim4(7));
    expectValues(output.data(), {0, 0, 1, 2, 3, 0, 0});
}

TEST(PadTest, AsymmetricVectorPadding) {
    Tensor input({1.0f, 2.0f, 3.0f});

    Tensor output = input.pad({1, 0, 0, 0}, {3, 0, 0, 0});

    EXPECT_EQ(output.shape(), af::dim4(7));
    expectValues(output.data(), {0, 1, 2, 3, 0, 0, 0});
}

TEST(PadTest, MultidimensionalPadding) {
    const float values[]{1, 2, 3, 4};
    Tensor input(af::array(2, 2, values));

    Tensor output = input.pad({1, 1, 0, 0}, {1, 1, 0, 0});

    EXPECT_EQ(output.shape(), af::dim4(4, 4));

    expectValues(output.data(), {0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0});
}

TEST(PadTest, PadsImageSpatialDimensions) {
    const float values[]{1, 2, 3, 4, 5, 6};
    Tensor input(af::array(1, 2, 3, values));

    Tensor output = input.pad({0, 1, 1, 0}, {0, 0, 2, 0});

    EXPECT_EQ(output.shape(), af::dim4(1, 3, 6));

    expectValues(output.data(), {0, 0, 0, 0, 1, 2, 0, 3, 4, 0, 5, 6, 0, 0, 0, 0, 0, 0});
}

TEST(PadTest, PadsTrailingSingletonDimension) {
    Tensor input({1.0f, 2.0f});

    Tensor output = input.pad({0, 0, 0, 1}, {0, 0, 0, 1});

    EXPECT_EQ(output.shape(), af::dim4(2, 1, 1, 3));

    expectValues(output.data(), {0, 0, 1, 2, 0, 0});
}

TEST(PadTest, ZeroPaddingPreservesInput) {
    Tensor input({1.0f, 2.0f, 3.0f});

    Tensor output = input.pad({0, 0, 0, 0}, {0, 0, 0, 0});

    EXPECT_EQ(output.shape(), input.shape());
    expectValues(output.data(), {1, 2, 3});
}

TEST(PadTest, PreservesIntegerType) {
    const unsigned char values[]{1, 2, 3};
    Tensor input(af::array(3, values));

    Tensor output = input.pad({1, 0, 0, 0}, {1, 0, 0, 0});

    ASSERT_EQ(output.data().type(), u8);
    ASSERT_EQ(output.data().elements(), 5);

    unsigned char actual[5];
    output.data().host(actual);

    EXPECT_EQ(actual[0], 0);
    EXPECT_EQ(actual[1], 1);
    EXPECT_EQ(actual[2], 2);
    EXPECT_EQ(actual[3], 3);
    EXPECT_EQ(actual[4], 0);
}

TEST(PadTest, DoesNotModifyInput) {
    Tensor input({1.0f, 2.0f, 3.0f});

    Tensor output = input.pad({1, 0, 0, 0}, {2, 0, 0, 0});

    EXPECT_EQ(input.shape(), af::dim4(3));
    expectValues(input.data(), {1, 2, 3});
    expectValues(output.data(), {0, 1, 2, 3, 0, 0});
}

TEST(PadTest, RejectsNegativePadding) {
    Tensor input({1.0f, 2.0f});

    for (int dim = 0; dim < 4; ++dim) {
        af::dim4 zero(0, 0, 0, 0);
        af::dim4 negative(0, 0, 0, 0);

        negative[dim] = -1;

        EXPECT_THROW(input.pad(negative, zero), std::invalid_argument);

        EXPECT_THROW(input.pad(zero, negative), std::invalid_argument);
    }
}

TEST(PadTest, RejectsEmptyInput) {
    Tensor input;

    EXPECT_THROW(input.pad({1, 0, 0, 0}, {1, 0, 0, 0}), std::invalid_argument);
}

TEST(PadTest, BackwardExtractsOriginalRegion) {
    const float values[]{1, 2, 3};
    Tensor input(af::array(3, values), true);

    Tensor weights({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    Tensor output = input.pad({2, 0, 0, 0}, {1, 0, 0, 0});

    Tensor loss = (output * weights).sum();
    loss.backward();

    EXPECT_EQ(input.grad().dims(), input.shape());
    expectValues(input.grad(), {3, 4, 5});
}

TEST(PadTest, BackwardExtractsMultidimensionalRegion) {
    const float values[]{1, 2, 3, 4};
    Tensor input(af::array(2, 2, values), true);

    const float weight_values[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    Tensor weights(af::array(4, 3, weight_values));

    Tensor output = input.pad({1, 1, 0, 0}, {1, 0, 0, 0});

    ASSERT_EQ(output.shape(), weights.shape());

    Tensor loss = (output * weights).sum();
    loss.backward();

    EXPECT_EQ(input.grad().dims(), input.shape());
    expectValues(input.grad(), {6, 7, 10, 11});
}