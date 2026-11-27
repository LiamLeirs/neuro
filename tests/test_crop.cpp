#include "neuro/Tensor.hpp"

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace neuro;

namespace {

af::array sequentialValues(const af::dim4& dims) {
    return af::moddims(af::range(af::dim4(dims.elements()), 0, f32), dims);
}

void expectValues(const af::array& array, const std::vector<float>& expected) {
    ASSERT_EQ(array.elements(), expected.size());

    std::vector<float> actual(expected.size());
    array.host(actual.data());

    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(actual[i], expected[i]) << "Index: " << i;
    }
}

} // namespace

TEST(SliceNDTest, SlicesVector) {
    Tensor input({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    Tensor output = input.slice({1, 0, 0, 0}, {4, 1, 1, 1});

    EXPECT_EQ(output.shape(), af::dim4(3));
    expectValues(output.data(), {2, 3, 4});
}

TEST(SliceNDTest, SlicesMatrix) {
    Tensor input(sequentialValues(af::dim4(4, 3)));

    Tensor output = input.slice({1, 1, 0, 0}, {3, 3, 1, 1});

    EXPECT_EQ(output.shape(), af::dim4(2, 2));
    expectValues(output.data(), {5, 6, 9, 10});
}

TEST(SliceNDTest, SlicesAllFourDimensions) {
    Tensor input(sequentialValues(af::dim4(3, 3, 3, 3)));

    Tensor output = input.slice({1, 1, 1, 1}, {3, 3, 3, 3});

    EXPECT_EQ(output.shape(), af::dim4(2, 2, 2, 2));

    expectValues(output.data(), {40, 41, 43, 44, 49, 50, 52, 53, 67, 68, 70, 71, 76, 77, 79, 80});
}

TEST(SliceNDTest, FullRangePreservesValues) {
    Tensor input(sequentialValues(af::dim4(2, 3)));

    Tensor output = input.slice({0, 0, 0, 0}, input.shape());

    EXPECT_EQ(output.shape(), input.shape());
    expectValues(output.data(), {0, 1, 2, 3, 4, 5});
}

TEST(SliceNDTest, SlicesSingleElement) {
    Tensor input(sequentialValues(af::dim4(3, 3)));

    Tensor output = input.slice({2, 1, 0, 0}, {3, 2, 1, 1});

    EXPECT_EQ(output.shape(), af::dim4(1));
    expectValues(output.data(), {5});
}

TEST(SliceNDTest, RejectsInvalidRanges) {
    Tensor input(af::constant(1, af::dim4(3, 3, 3, 3)));

    for (int dim = 0; dim < 4; ++dim) {
        af::dim4 start(0, 0, 0, 0);
        af::dim4 end(3, 3, 3, 3);

        end[dim] = 0;
        EXPECT_THROW(input.slice(start, end), std::out_of_range);

        start[dim] = 2;
        end[dim] = 1;
        EXPECT_THROW(input.slice(start, end), std::out_of_range);
    }
}

TEST(SliceNDTest, RejectsOutOfBoundsIndices) {
    Tensor input(af::constant(1, af::dim4(3, 3, 3, 3)));

    for (int dim = 0; dim < 4; ++dim) {
        af::dim4 start(0, 0, 0, 0);
        af::dim4 end(3, 3, 3, 3);

        start[dim] = -1;
        EXPECT_THROW(input.slice(start, end), std::out_of_range);

        start[dim] = 0;
        end[dim] = 4;
        EXPECT_THROW(input.slice(start, end), std::out_of_range);
    }
}

TEST(SliceNDTest, RejectsEmptyInput) {
    Tensor input;

    EXPECT_THROW(input.slice({0, 0, 0, 0}, {1, 1, 1, 1}), std::invalid_argument);
}

TEST(SliceNDTest, BackwardUpdatesSelectedRegion) {
    Tensor input(sequentialValues(af::dim4(4, 3)), true);

    const float values[]{2, 3, 5, 7};
    Tensor weights(af::array(2, 2, values));

    Tensor output = input.slice({1, 1, 0, 0}, {3, 3, 1, 1});

    Tensor loss = (output * weights).sum();
    loss.backward();

    EXPECT_EQ(input.grad().dims(), input.shape());

    expectValues(input.grad(), {0, 0, 0, 0, 0, 2, 3, 0, 0, 5, 7, 0});
}

TEST(SliceNDTest, BackwardAcrossFourDimensions) {
    Tensor input(af::constant(1, af::dim4(2, 2, 2, 2)), true);

    Tensor output = input.slice({1, 0, 1, 0}, {2, 2, 2, 2});

    const float values[]{2, 3, 5, 7};
    Tensor weights(af::array(af::dim4(1, 2, 1, 2), values));

    Tensor loss = (output * weights).sum();
    loss.backward();

    EXPECT_EQ(input.grad().dims(), input.shape());

    expectValues(input.grad(), {0, 0, 0, 0, 0, 2, 0, 3, 0, 0, 0, 0, 0, 5, 0, 7});
}

TEST(CropTest, CropsNonSquareImage) {
    Tensor input(sequentialValues(af::dim4(3, 4, 5)));

    Tensor output = input.crop({0, 1, 2, 0}, {3, 2, 3, 1});

    EXPECT_EQ(output.shape(), af::dim4(3, 2, 3));

    expectValues(output.data(),
                 {27, 28, 29, 30, 31, 32, 39, 40, 41, 42, 43, 44, 51, 52, 53, 54, 55, 56});
}

TEST(CropTest, CropsSingletonDimensions) {
    Tensor input(sequentialValues(af::dim4(1, 4, 5)));

    Tensor output = input.crop({0, 1, 2, 0}, {1, 2, 2, 1});

    EXPECT_EQ(output.shape(), af::dim4(1, 2, 2));
    expectValues(output.data(), {9, 10, 13, 14});
}

TEST(CropTest, PreservesIntegerType) {
    const unsigned char values[]{1, 2, 3, 4, 5};
    Tensor input(af::array(5, values));

    Tensor output = input.crop({1, 0, 0, 0}, {3, 1, 1, 1});

    ASSERT_EQ(output.data().type(), u8);
    ASSERT_EQ(output.data().elements(), 3);

    unsigned char actual[3];
    output.data().host(actual);

    EXPECT_EQ(actual[0], 2);
    EXPECT_EQ(actual[1], 3);
    EXPECT_EQ(actual[2], 4);
}

TEST(CropTest, DoesNotModifyInput) {
    Tensor input({1.0f, 2.0f, 3.0f, 4.0f});

    Tensor output = input.crop({1, 0, 0, 0}, {2, 1, 1, 1});

    EXPECT_EQ(input.shape(), af::dim4(4));
    expectValues(input.data(), {1, 2, 3, 4});
    expectValues(output.data(), {2, 3});
}

TEST(CropTest, RejectsNonPositiveSizes) {
    Tensor input(af::constant(1, af::dim4(3, 3, 3, 3)));

    for (int dim = 0; dim < 4; ++dim) {
        af::dim4 size(1, 1, 1, 1);

        size[dim] = 0;
        EXPECT_THROW(input.crop({0, 0, 0, 0}, size), std::invalid_argument);

        size[dim] = -1;
        EXPECT_THROW(input.crop({0, 0, 0, 0}, size), std::invalid_argument);
    }
}

TEST(CropTest, RejectsOutOfBoundsRegions) {
    Tensor input(af::constant(1, af::dim4(3, 3, 3, 3)));

    for (int dim = 0; dim < 4; ++dim) {
        af::dim4 start(0, 0, 0, 0);
        af::dim4 size(1, 1, 1, 1);

        start[dim] = -1;
        EXPECT_THROW(input.crop(start, size), std::out_of_range);

        start[dim] = 3;
        EXPECT_THROW(input.crop(start, size), std::out_of_range);

        start[dim] = 2;
        size[dim] = 2;
        EXPECT_THROW(input.crop(start, size), std::out_of_range);
    }
}

TEST(CropTest, RejectsOverflowingRegion) {
    Tensor input({1.0f, 2.0f, 3.0f});

    EXPECT_THROW(input.crop({1, 0, 0, 0}, {std::numeric_limits<dim_t>::max(), 1, 1, 1}),
                 std::out_of_range);
}

TEST(CropTest, RejectsEmptyInput) {
    Tensor input;

    EXPECT_THROW(input.crop({0, 0, 0, 0}, {1, 1, 1, 1}), std::invalid_argument);
}

TEST(CropTest, BackwardAccumulatesOverlappingCrops) {
    Tensor input({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, true);

    Tensor first = input.crop({0, 0, 0, 0}, {3, 1, 1, 1});
    Tensor second = input.crop({2, 0, 0, 0}, {3, 1, 1, 1});

    Tensor loss = first.sum() * 2.0 + second.sum() * 3.0;
    loss.backward();

    expectValues(input.grad(), {2, 2, 5, 3, 3});
}

TEST(CropTest, BackwardThroughPadding) {
    Tensor input({1.0f, 2.0f, 3.0f}, true);

    Tensor padded = input.pad({1, 0, 0, 0}, {1, 0, 0, 0});
    Tensor cropped = padded.crop({0, 0, 0, 0}, {3, 1, 1, 1});

    expectValues(cropped.data(), {0, 1, 2});

    Tensor weights({2.0f, 3.0f, 5.0f});

    Tensor loss = (cropped * weights).sum();
    loss.backward();

    expectValues(input.grad(), {3, 5, 0});
}