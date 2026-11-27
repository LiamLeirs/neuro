#include "neuro/Tensor.hpp"
#include "neuro/data/transforms/RandomCrop.hpp"

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <numeric>
#include <set>
#include <stdexcept>
#include <vector>

using namespace neuro;
using namespace neuro::transforms;

namespace {

Tensor makeInput(const af::dim4& dims, bool requires_grad = false) {
    std::vector<float> values(dims.elements());
    std::iota(values.begin(), values.end(), 1.0f);

    return Tensor(af::array(dims, values.data()), requires_grad);
}

std::vector<float> hostValues(const af::array& array) {
    std::vector<float> values(array.elements());
    array.host(values.data());
    return values;
}

bool isValidCrop(const Tensor& input, const Tensor& output) {
    const auto dims = input.shape();
    const auto crop_dims = output.shape();

    if (dims[0] != crop_dims[0] || crop_dims[1] > dims[1] || crop_dims[2] > dims[2]) {
        return false;
    }

    const auto actual = hostValues(output.data());

    for (dim_t y = 0; y <= dims[1] - crop_dims[1]; ++y) {
        for (dim_t x = 0; x <= dims[2] - crop_dims[2]; ++x) {
            Tensor candidate = input.crop({0, y, x, 0}, crop_dims);

            if (hostValues(candidate.data()) == actual) {
                return true;
            }
        }
    }

    return false;
}

} // namespace

TEST(RandomCropTest, RejectsInvalidConstructorArguments) {
    EXPECT_THROW(RandomCrop(0, 3), std::invalid_argument);
    EXPECT_THROW(RandomCrop(3, 0), std::invalid_argument);
    EXPECT_THROW(RandomCrop(-1, 3), std::invalid_argument);
    EXPECT_THROW(RandomCrop(3, -1), std::invalid_argument);
    EXPECT_THROW(RandomCrop(3, 3, -1), std::invalid_argument);
}

TEST(RandomCropTest, SupportsBatchForwarding) {
    RandomCrop crop(3, 3);

    EXPECT_TRUE(crop.supportsBatch());
}

TEST(RandomCropTest, SingleSampleOutputShape) {
    Tensor input = makeInput({3, 8, 10});

    RandomCrop crop(4, 6);
    Tensor output = crop(input);

    EXPECT_EQ(output.shape(), af::dim4(3, 4, 6));
}

TEST(RandomCropTest, BatchOutputShape) {
    Tensor input = makeInput({8, 3, 8, 10});

    RandomCrop crop(4, 6);
    Tensor output = crop.forwardBatch(input);

    EXPECT_EQ(output.shape(), af::dim4(8, 3, 4, 6));
}

TEST(RandomCropTest, IdentitySingleSample) {
    Tensor input = makeInput({3, 4, 5});

    RandomCrop crop(4, 5);
    Tensor output = crop(input);

    EXPECT_EQ(output.shape(), input.shape());
    EXPECT_EQ(hostValues(output.data()), hostValues(input.data()));
}

TEST(RandomCropTest, IdentityBatch) {
    Tensor input = makeInput({4, 3, 4, 5});

    RandomCrop crop(4, 5);
    Tensor output = crop.forwardBatch(input);

    EXPECT_EQ(output.shape(), input.shape());
    EXPECT_EQ(hostValues(output.data()), hostValues(input.data()));
}

TEST(RandomCropTest, AppliesZeroPaddingToSingleSample) {
    Tensor input = makeInput({3, 2, 3});

    RandomCrop crop(4, 5, 1);
    Tensor output = crop(input);

    Tensor expected = input.pad({0, 1, 1, 0}, {0, 1, 1, 0});

    EXPECT_EQ(output.shape(), af::dim4(3, 4, 5));
    EXPECT_EQ(hostValues(output.data()), hostValues(expected.data()));
}

TEST(RandomCropTest, AppliesZeroPaddingToBatch) {
    Tensor input = makeInput({4, 3, 2, 3});

    RandomCrop crop(4, 5, 1);
    Tensor output = crop.forwardBatch(input);

    Tensor expected = input.pad({0, 0, 1, 1}, {0, 0, 1, 1});

    EXPECT_EQ(output.shape(), af::dim4(4, 3, 4, 5));
    EXPECT_EQ(hostValues(output.data()), hostValues(expected.data()));
}

TEST(RandomCropTest, ProducesValidSingleSampleCrops) {
    Tensor input = makeInput({3, 5, 6});

    RandomCrop crop(3, 4);

    for (int i = 0; i < 10; ++i) {
        Tensor output = crop(input);

        ASSERT_TRUE(isValidCrop(input, output));
    }
}

TEST(RandomCropTest, ProducesValidPaddedCrops) {
    Tensor input = makeInput({3, 4, 5});

    RandomCrop crop(4, 5, 2);

    Tensor padded = input.pad({0, 2, 2, 0}, {0, 2, 2, 0});

    for (int i = 0; i < 10; ++i) {
        Tensor output = crop(input);

        ASSERT_TRUE(isValidCrop(padded, output));
    }
}

TEST(RandomCropTest, HandlesFixedVerticalOffset) {
    Tensor input = makeInput({3, 4, 6});

    RandomCrop crop(4, 3);

    for (int i = 0; i < 5; ++i) {
        Tensor output = crop(input);

        EXPECT_EQ(output.shape(), af::dim4(3, 4, 3));
        ASSERT_TRUE(isValidCrop(input, output));
    }
}

TEST(RandomCropTest, HandlesFixedHorizontalOffset) {
    Tensor input = makeInput({3, 6, 4});

    RandomCrop crop(3, 4);

    for (int i = 0; i < 5; ++i) {
        Tensor output = crop(input);

        EXPECT_EQ(output.shape(), af::dim4(3, 3, 4));
        ASSERT_TRUE(isValidCrop(input, output));
    }
}

TEST(RandomCropTest, ProducesValidIndependentBatchCrops) {
    constexpr int batch_size = 8;
    constexpr int channels = 3;

    Tensor input = makeInput({batch_size, channels, 5, 6});

    RandomCrop crop(3, 4, 1);
    Tensor output = crop.forwardBatch(input);

    ASSERT_EQ(output.shape(), af::dim4(batch_size, channels, 3, 4));

    for (int n = 0; n < batch_size; ++n) {
        Tensor sample = input.crop({n, 0, 0, 0}, {1, channels, 5, 6}).reshape({channels, 5, 6});

        Tensor padded = sample.pad({0, 1, 1, 0}, {0, 1, 1, 0});

        Tensor cropped_sample =
            output.crop({n, 0, 0, 0}, {1, channels, 3, 4}).reshape({channels, 3, 4});

        ASSERT_TRUE(isValidCrop(padded, cropped_sample)) << "Invalid crop for sample " << n;
    }
}

TEST(RandomCropTest, SamplesDifferentOffsetsAcrossBatch) {
    constexpr int batch_size = 64;

    const auto image = af::moddims(af::range(af::dim4(64), 0, f32), af::dim4(1, 1, 8, 8));

    Tensor input(af::tile(image, batch_size, 1, 1, 1));

    RandomCrop crop(4, 4);
    Tensor output = crop.forwardBatch(input);

    Tensor first_pixels = output.crop({0, 0, 0, 0}, {batch_size, 1, 1, 1});

    const auto values = hostValues(first_pixels.data());

    const std::set<float> unique_values(values.begin(), values.end());

    EXPECT_GT(unique_values.size(), 1u);
}

TEST(RandomCropTest, SamplesDifferentOffsetsAcrossCalls) {
    Tensor input = makeInput({1, 8, 8});

    RandomCrop crop(4, 4);

    std::set<float> first_pixels;

    for (int i = 0; i < 20; ++i) {
        Tensor output = crop(input);

        first_pixels.insert(output.data()(0, 0, 0).scalar<float>());
    }

    EXPECT_GT(first_pixels.size(), 1u);
}

TEST(RandomCropTest, PreservesIntegerDtype) {
    Tensor input(af::constant(7, af::dim4(3, 5, 6), u8));

    RandomCrop crop(3, 4, 1);
    Tensor output = crop(input);

    EXPECT_EQ(output.data().type(), u8);

    Tensor batch(af::constant(7, af::dim4(4, 3, 5, 6), u8));

    Tensor batch_output = crop.forwardBatch(batch);

    EXPECT_EQ(batch_output.data().type(), u8);
}

TEST(RandomCropTest, DoesNotModifyInput) {
    Tensor input = makeInput({4, 3, 5, 6});

    const auto original = hostValues(input.data());

    RandomCrop crop(3, 4, 1);
    Tensor output = crop.forwardBatch(input);

    EXPECT_EQ(input.shape(), af::dim4(4, 3, 5, 6));
    EXPECT_EQ(hostValues(input.data()), original);
}

TEST(RandomCropTest, RejectsEmptyInput) {
    Tensor input;
    RandomCrop crop(3, 3);

    EXPECT_THROW(crop(input), std::invalid_argument);

    EXPECT_THROW(crop.forwardBatch(input), std::invalid_argument);
}

TEST(RandomCropTest, RejectsOversizedCrop) {
    Tensor sample = makeInput({3, 4, 5});
    Tensor batch = makeInput({4, 3, 4, 5});

    RandomCrop too_tall(7, 3, 1);
    RandomCrop too_wide(3, 8, 1);

    EXPECT_THROW(too_tall(sample), std::out_of_range);
    EXPECT_THROW(too_wide(sample), std::out_of_range);

    EXPECT_THROW(too_tall.forwardBatch(batch), std::out_of_range);

    EXPECT_THROW(too_wide.forwardBatch(batch), std::out_of_range);
}

TEST(RandomCropTest, RejectsFourDimensionalSingleSample) {
    Tensor input = makeInput({4, 3, 5, 6});

    RandomCrop crop(3, 3);

    EXPECT_THROW(crop(input), std::invalid_argument);
}

TEST(RandomCropTest, SingleSampleBackward) {
    Tensor input = makeInput({3, 4, 5}, true);

    RandomCrop crop(3, 3, 1);
    Tensor output = crop(input);

    ASSERT_TRUE(output.requires_grad());

    const auto selected = hostValues(output.data());

    Tensor loss = output.sum();
    loss.backward();

    std::vector<float> expected(input.data().elements(), 0.0f);

    for (float value : selected) {
        if (value != 0.0f) {
            expected[static_cast<std::size_t>(value - 1)] += 1.0f;
        }
    }

    EXPECT_EQ(input.grad().dims(), input.shape());
    EXPECT_EQ(hostValues(input.grad()), expected);
}

TEST(RandomCropTest, BatchWeightedBackward) {
    Tensor input = makeInput({4, 3, 4, 5}, true);

    RandomCrop crop(3, 3, 1);
    Tensor output = crop.forwardBatch(input);

    ASSERT_TRUE(output.requires_grad());

    const auto selected = hostValues(output.data());

    std::vector<float> weights(output.data().elements());
    std::iota(weights.begin(), weights.end(), 1.0f);

    Tensor weight_tensor(af::array(output.shape(), weights.data()));

    Tensor loss = (output * weight_tensor).sum();
    loss.backward();

    std::vector<float> expected(input.data().elements(), 0.0f);

    for (std::size_t i = 0; i < selected.size(); ++i) {
        if (selected[i] != 0.0f) {
            expected[static_cast<std::size_t>(selected[i] - 1)] += weights[i];
        }
    }

    EXPECT_EQ(input.grad().dims(), input.shape());
    EXPECT_EQ(hostValues(input.grad()), expected);
}