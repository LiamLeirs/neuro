#include "neuro/Tensor.hpp"

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace neuro;

namespace {

Tensor makeInput(const af::dim4& dims, bool requires_grad = false) {
    std::vector<float> values(dims.elements());
    std::iota(values.begin(), values.end(), 0.0f);

    return Tensor(af::array(dims, values.data()), requires_grad);
}

Tensor makeOffsets(const std::vector<int>& values) {
    return Tensor(af::array(values.size(), values.data()));
}

void expectValues(const af::array& array, const std::vector<float>& expected) {
    ASSERT_EQ(array.elements(), expected.size());

    std::vector<float> actual(expected.size());
    array.host(actual.data());

    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(actual[i], expected[i]) << "Index: " << i;
    }
}

std::vector<float> referenceCrop(const Tensor& input, const std::vector<int>& offsets_y,
                                 const std::vector<int>& offsets_x, int height, int width) {

    const auto dims = input.shape();

    const dim_t N = dims[0];
    const dim_t C = dims[1];
    const dim_t H = dims[2];

    std::vector<float> source(dims.elements());
    input.data().host(source.data());

    std::vector<float> result;
    result.reserve(N * C * height * width);

    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            for (dim_t c = 0; c < C; ++c) {
                for (dim_t n = 0; n < N; ++n) {
                    const dim_t source_y = y + offsets_y[n];
                    const dim_t source_x = x + offsets_x[n];

                    const dim_t index = n + N * (c + C * (source_y + H * source_x));

                    result.push_back(source[index]);
                }
            }
        }
    }

    return result;
}

} // namespace

TEST(CropBatchTest, IndependentImageOffsets) {
    Tensor input = makeInput({3, 1, 4, 5});

    const std::vector<int> ys{0, 1, 2};
    const std::vector<int> xs{3, 0, 1};

    Tensor output = input.cropBatch(makeOffsets(ys), makeOffsets(xs), 2, 2);

    EXPECT_EQ(output.shape(), af::dim4(3, 1, 2, 2));
    expectValues(output.data(), referenceCrop(input, ys, xs, 2, 2));
}

TEST(CropBatchTest, MultipleChannelsAndNonSquareCrop) {
    Tensor input = makeInput({4, 3, 5, 7});

    const std::vector<int> ys{0, 2, 1, 3};
    const std::vector<int> xs{4, 0, 2, 1};

    Tensor output = input.cropBatch(makeOffsets(ys), makeOffsets(xs), 2, 3);

    EXPECT_EQ(output.shape(), af::dim4(4, 3, 2, 3));
    expectValues(output.data(), referenceCrop(input, ys, xs, 2, 3));
}

TEST(CropBatchTest, BoundaryOffsets) {
    Tensor input = makeInput({4, 2, 5, 6});

    const std::vector<int> ys{0, 0, 3, 3};
    const std::vector<int> xs{0, 3, 0, 3};

    Tensor output = input.cropBatch(makeOffsets(ys), makeOffsets(xs), 2, 3);

    EXPECT_EQ(output.shape(), af::dim4(4, 2, 2, 3));
    expectValues(output.data(), referenceCrop(input, ys, xs, 2, 3));
}

TEST(CropBatchTest, FullImageCrop) {
    Tensor input = makeInput({3, 2, 4, 5});

    Tensor offsets = makeOffsets({0, 0, 0});

    Tensor output = input.cropBatch(offsets, offsets, 4, 5);

    EXPECT_EQ(output.shape(), input.shape());

    std::vector<float> expected(input.data().elements());
    input.data().host(expected.data());

    expectValues(output.data(), expected);
}

TEST(CropBatchTest, SingleImageBatch) {
    Tensor input = makeInput({1, 3, 5, 6});

    Tensor output = input.cropBatch(makeOffsets({2}), makeOffsets({1}), 3, 4);

    Tensor expected = input.crop({0, 0, 2, 1}, {1, 3, 3, 4});

    EXPECT_EQ(output.shape(), expected.shape());

    std::vector<float> values(expected.data().elements());
    expected.data().host(values.data());

    expectValues(output.data(), values);
}

TEST(CropBatchTest, SinglePixelCrop) {
    Tensor input = makeInput({3, 2, 4, 5});

    const std::vector<int> ys{0, 3, 1};
    const std::vector<int> xs{4, 0, 2};

    Tensor output = input.cropBatch(makeOffsets(ys), makeOffsets(xs), 1, 1);

    EXPECT_EQ(output.shape(), af::dim4(3, 2, 1, 1));
    expectValues(output.data(), referenceCrop(input, ys, xs, 1, 1));
}

TEST(CropBatchTest, PreservesIntegerDtype) {
    const unsigned char values[]{1, 2, 3, 4, 5, 6, 7, 8};

    Tensor input(af::array(af::dim4(2, 1, 2, 2), values));

    Tensor output = input.cropBatch(makeOffsets({0, 1}), makeOffsets({0, 1}), 1, 1);

    ASSERT_EQ(output.data().type(), u8);
    ASSERT_EQ(output.shape(), af::dim4(2, 1, 1, 1));

    unsigned char actual[2];
    output.data().host(actual);

    EXPECT_EQ(actual[0], 1);
    EXPECT_EQ(actual[1], 8);
}

TEST(CropBatchTest, DoesNotModifyInput) {
    Tensor input = makeInput({2, 3, 4, 5});

    std::vector<float> original(input.data().elements());
    input.data().host(original.data());

    Tensor output = input.cropBatch(makeOffsets({0, 1}), makeOffsets({1, 2}), 2, 2);

    EXPECT_EQ(input.shape(), af::dim4(2, 3, 4, 5));
    expectValues(input.data(), original);
}

TEST(CropBatchTest, RejectsEmptyTensors) {
    Tensor input = makeInput({2, 1, 4, 4});
    Tensor offsets = makeOffsets({0, 0});
    Tensor empty;

    EXPECT_THROW(empty.cropBatch(offsets, offsets, 2, 2), std::invalid_argument);

    EXPECT_THROW(input.cropBatch(empty, offsets, 2, 2), std::invalid_argument);

    EXPECT_THROW(input.cropBatch(offsets, empty, 2, 2), std::invalid_argument);
}

TEST(CropBatchTest, RejectsInvalidOffsetShapes) {
    Tensor input = makeInput({3, 1, 4, 4});
    Tensor valid = makeOffsets({0, 0, 0});

    Tensor too_short = makeOffsets({0, 0});
    Tensor too_long = makeOffsets({0, 0, 0, 0});
    Tensor wrong_shape(af::constant(0, af::dim4(3, 2), s32));

    for (const auto& invalid : {too_short, too_long, wrong_shape}) {
        EXPECT_THROW(input.cropBatch(invalid, valid, 2, 2), std::invalid_argument);

        EXPECT_THROW(input.cropBatch(valid, invalid, 2, 2), std::invalid_argument);
    }
}

TEST(CropBatchTest, RejectsInvalidOffsetDtypes) {
    Tensor input = makeInput({2, 1, 4, 4});
    Tensor valid = makeOffsets({0, 0});

    for (auto dtype : {f32, f64, u32, s64}) {
        Tensor invalid(af::constant(0, af::dim4(2), dtype));

        EXPECT_THROW(input.cropBatch(invalid, valid, 2, 2), std::invalid_argument);

        EXPECT_THROW(input.cropBatch(valid, invalid, 2, 2), std::invalid_argument);
    }
}

TEST(CropBatchTest, RejectsNonPositiveCropDimensions) {
    Tensor input = makeInput({2, 1, 4, 4});
    Tensor offsets = makeOffsets({0, 0});

    EXPECT_THROW(input.cropBatch(offsets, offsets, 0, 2), std::invalid_argument);

    EXPECT_THROW(input.cropBatch(offsets, offsets, 2, 0), std::invalid_argument);

    EXPECT_THROW(input.cropBatch(offsets, offsets, -1, 2), std::invalid_argument);

    EXPECT_THROW(input.cropBatch(offsets, offsets, 2, -1), std::invalid_argument);
}

TEST(CropBatchTest, RejectsOversizedCrop) {
    Tensor input = makeInput({2, 1, 4, 5});
    Tensor offsets = makeOffsets({0, 0});

    EXPECT_THROW(input.cropBatch(offsets, offsets, 5, 3), std::out_of_range);

    EXPECT_THROW(input.cropBatch(offsets, offsets, 3, 6), std::out_of_range);
}

TEST(CropBatchTest, RejectsOutOfBoundsOffsets) {
    Tensor input = makeInput({2, 1, 4, 5});
    Tensor valid = makeOffsets({0, 0});

    EXPECT_THROW(input.cropBatch(makeOffsets({0, -1}), valid, 2, 2), std::out_of_range);

    EXPECT_THROW(input.cropBatch(valid, makeOffsets({-1, 0}), 2, 2), std::out_of_range);

    EXPECT_THROW(input.cropBatch(makeOffsets({0, 3}), valid, 2, 2), std::out_of_range);

    EXPECT_THROW(input.cropBatch(valid, makeOffsets({4, 0}), 2, 2), std::out_of_range);
}

TEST(CropBatchTest, DetachedInputProducesDetachedOutput) {
    Tensor input = makeInput({2, 1, 4, 4});
    Tensor offsets = makeOffsets({0, 1});

    Tensor output = input.cropBatch(offsets, offsets, 2, 2);

    EXPECT_FALSE(output.requires_grad());
}

TEST(CropBatchTest, WeightedBackward) {
    Tensor input = makeInput({3, 2, 4, 5}, true);

    const std::vector<int> ys{0, 1, 2};
    const std::vector<int> xs{3, 0, 1};

    constexpr int height = 2;
    constexpr int width = 2;

    Tensor output = input.cropBatch(makeOffsets(ys), makeOffsets(xs), height, width);

    ASSERT_TRUE(output.requires_grad());

    const auto dims = input.shape();

    const dim_t N = dims[0];
    const dim_t C = dims[1];
    const dim_t H = dims[2];

    std::vector<float> weights(output.data().elements());
    std::iota(weights.begin(), weights.end(), 1.0f);

    Tensor weight_tensor(af::array(output.shape(), weights.data()));

    Tensor loss = (output * weight_tensor).sum();
    loss.backward();

    std::vector<float> expected(dims.elements(), 0.0f);

    std::size_t output_index = 0;

    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            for (dim_t c = 0; c < C; ++c) {
                for (dim_t n = 0; n < N; ++n) {
                    const dim_t input_index = n + N * (c + C * (y + ys[n] + H * (x + xs[n])));

                    expected[input_index] += weights[output_index++];
                }
            }
        }
    }

    EXPECT_EQ(input.grad().dims(), input.shape());
    expectValues(input.grad(), expected);
}

TEST(CropBatchTest, AccumulatesOverlappingCropGradients) {
    Tensor input = makeInput({2, 1, 1, 4}, true);

    Tensor ys = makeOffsets({0, 0});

    Tensor first = input.cropBatch(ys, makeOffsets({0, 1}), 1, 3);

    Tensor second = input.cropBatch(ys, makeOffsets({1, 0}), 1, 3);

    Tensor loss = first.sum() * 2.0 + second.sum() * 3.0;
    loss.backward();

    expectValues(input.grad(), {2, 3, 5, 5, 5, 5, 3, 2});
}

TEST(CropBatchTest, AccumulatesWithOtherComputationalPaths) {
    Tensor input = makeInput({2, 1, 1, 4}, true);

    Tensor cropped = input.cropBatch(makeOffsets({0, 0}), makeOffsets({1, 2}), 1, 2);

    Tensor loss = input.sum() + cropped.sum() * 2.0;
    loss.backward();

    expectValues(input.grad(), {1, 1, 3, 1, 3, 3, 1, 3});
}

TEST(CropBatchTest, BackwardThroughPadding) {
    Tensor input = makeInput({2, 1, 2, 2}, true);

    Tensor padded = input.pad({0, 0, 1, 1}, {0, 0, 1, 1});

    Tensor output = padded.cropBatch(makeOffsets({0, 2}), makeOffsets({0, 2}), 2, 2);

    EXPECT_EQ(output.shape(), af::dim4(2, 1, 2, 2));

    expectValues(output.data(), {0, 7, 0, 0, 0, 0, 0, 0});

    Tensor loss = output.sum();
    loss.backward();

    EXPECT_EQ(input.grad().dims(), input.shape());

    expectValues(input.grad(), {1, 0, 0, 0, 0, 0, 0, 1});
}