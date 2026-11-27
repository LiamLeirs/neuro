#include "neuro/Tensor.hpp"
#include "neuro/data/transforms/Normalize.hpp"

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace neuro;
using namespace neuro::transforms;

namespace {

constexpr double kTolerance = 1e-5;

void expectValues(const Tensor& tensor, const std::vector<float>& expected) {
    ASSERT_EQ(tensor.data().elements(), expected.size());

    std::vector<float> actual(expected.size());
    tensor.data().host(actual.data());

    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], kTolerance) << "Index: " << i;
    }
}

} // namespace

TEST(NormalizeTest, RejectsEmptyStatistics) {
    EXPECT_THROW(Normalize({}, {}), std::invalid_argument);
    EXPECT_THROW(Normalize({0.5f}, {}), std::invalid_argument);
    EXPECT_THROW(Normalize({}, {0.5f}), std::invalid_argument);
}

TEST(NormalizeTest, RejectsMismatchedStatistics) {
    EXPECT_THROW(Normalize({0.5f, 0.3f}, {0.2f}), std::invalid_argument);
}

TEST(NormalizeTest, RejectsNonPositiveStandardDeviation) {
    EXPECT_THROW(Normalize({0.5f}, {0.0f}), std::invalid_argument);

    EXPECT_THROW(Normalize({0.5f}, {-0.2f}), std::invalid_argument);
}

TEST(NormalizeTest, RejectsNonFiniteStatistics) {
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();

    EXPECT_THROW(Normalize({inf}, {1.0f}), std::invalid_argument);
    EXPECT_THROW(Normalize({nan}, {1.0f}), std::invalid_argument);
    EXPECT_THROW(Normalize({0.0f}, {inf}), std::invalid_argument);
    EXPECT_THROW(Normalize({0.0f}, {nan}), std::invalid_argument);
}

TEST(NormalizeTest, NormalizesFeatureVector) {
    Normalize transform({1.0f, 2.0f, 3.0f}, {2.0f, 4.0f, 5.0f});

    Tensor input({3.0f, 6.0f, -2.0f});

    Tensor output = transform(input);

    EXPECT_EQ(output.shape(), input.shape());
    expectValues(output, {1.0f, 1.0f, -1.0f});
}

TEST(NormalizeTest, NormalizesGrayscaleImage) {
    Normalize transform({0.5f}, {0.25f});

    const float values[]{0.0f, 0.25f, 0.5f, 1.0f};

    Tensor input(af::array(1, 2, 2, values));

    Tensor output = transform(input);

    EXPECT_EQ(output.shape(), input.shape());
    expectValues(output, {-2.0f, -1.0f, 0.0f, 2.0f});
}

TEST(NormalizeTest, NormalizesRGBChannelsIndependently) {
    Normalize transform({1.0f, 2.0f, 3.0f}, {1.0f, 2.0f, 4.0f});

    // CHW layout: channels are contiguous in ArrayFire.
    const float values[]{2.0f, 6.0f, 15.0f, 0.0f, 2.0f, 7.0f, 1.0f, 0.0f, 3.0f, 4.0f, 8.0f, -1.0f};

    Tensor input(af::array(3, 2, 2, values));

    Tensor output = transform(input);

    EXPECT_EQ(output.shape(), input.shape());

    expectValues(output,
                 {1.0f, 2.0f, 3.0f, -1.0f, 0.0f, 1.0f, 0.0f, -1.0f, 0.0f, 3.0f, 3.0f, -1.0f});
}

TEST(NormalizeTest, SupportsBatching) {
    Normalize transform({0.5f}, {0.25f});

    EXPECT_TRUE(transform.supportsBatch());
}

TEST(NormalizeTest, NormalizesBatchedFeatureVectors) {
    Normalize transform({1.0f, 2.0f, 3.0f}, {1.0f, 2.0f, 4.0f});

    // Shape: (N=2, C=3).
    const float values[]{2.0f, 0.0f, 6.0f, 2.0f, 15.0f, 7.0f};

    Tensor input(af::array(2, 3, values));

    Tensor output = transform.forwardBatch(input);

    EXPECT_EQ(output.shape(), input.shape());

    expectValues(output, {1.0f, -1.0f, 2.0f, 0.0f, 3.0f, 1.0f});
}

TEST(NormalizeTest, NormalizesBatchedRGBImages) {
    Normalize transform({1.0f, 2.0f, 3.0f}, {1.0f, 2.0f, 4.0f});

    constexpr int n = 2;
    constexpr int c = 3;
    constexpr int h = 2;
    constexpr int w = 3;

    const float means[]{1.0f, 2.0f, 3.0f};
    const float stds[]{1.0f, 2.0f, 4.0f};

    std::vector<float> values(n * c * h * w);
    std::vector<float> expected(values.size());

    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < h; ++y) {
            for (int channel = 0; channel < c; ++channel) {
                for (int b = 0; b < n; ++b) {
                    const int index = b + n * (channel + c * (y + h * x));

                    values[index] = static_cast<float>(index);

                    expected[index] = (values[index] - means[channel]) / stds[channel];
                }
            }
        }
    }

    Tensor input(af::array(n, c, h, w, values.data()));

    Tensor output = transform.forwardBatch(input);

    EXPECT_EQ(output.shape(), input.shape());
    expectValues(output, expected);
}

TEST(NormalizeTest, BatchMatchesIndividualNormalization) {
    Normalize transform({1.0f, 2.0f, 3.0f}, {1.0f, 2.0f, 4.0f});

    const float first_values[]{2.0f, 6.0f, 15.0f, 0.0f, 2.0f, 7.0f};

    const float second_values[]{1.0f, 0.0f, 3.0f, 4.0f, 8.0f, -1.0f};

    Tensor first(af::array(3, 1, 2, first_values));
    Tensor second(af::array(3, 1, 2, second_values));

    Tensor batch = Tensor::stack(std::vector<Tensor>{first, second}, 0);

    Tensor expected = Tensor::stack(std::vector<Tensor>{transform(first), transform(second)}, 0);

    Tensor actual = transform.forwardBatch(batch);

    EXPECT_EQ(actual.shape(), expected.shape());

    EXPECT_NEAR(af::max<float>(af::abs(actual.data() - expected.data())), 0.0f, kTolerance);
}

TEST(NormalizeTest, RejectsIncorrectChannelCount) {
    Normalize transform({0.5f, 0.5f, 0.5f}, {0.2f, 0.2f, 0.2f});

    Tensor input(af::constant(1.0f, 1, 2, 2));

    EXPECT_THROW(transform(input), std::invalid_argument);
}

TEST(NormalizeTest, RejectsIncorrectBatchChannelCount) {
    Normalize transform({0.5f, 0.5f, 0.5f}, {0.2f, 0.2f, 0.2f});

    Tensor input(af::constant(1.0f, 4, 1, 2, 2));

    EXPECT_THROW(transform.forwardBatch(input), std::invalid_argument);
}

TEST(NormalizeTest, RejectsEmptyInput) {
    Normalize transform({0.5f}, {0.25f});

    Tensor input;

    EXPECT_THROW(transform(input), std::invalid_argument);
    EXPECT_THROW(transform.forwardBatch(input), std::invalid_argument);
}

TEST(NormalizeTest, RejectsIntegerInput) {
    Normalize transform({0.5f}, {0.25f});

    Tensor input(af::constant(128, 1, 2, 2, u8));
    Tensor batch(af::constant(128, 2, 1, 2, 2, u8));

    EXPECT_THROW(transform(input), std::invalid_argument);
    EXPECT_THROW(transform.forwardBatch(batch), std::invalid_argument);
}

TEST(NormalizeTest, PreservesFloat32Type) {
    Normalize transform({0.5f}, {0.25f});

    Tensor input({1.0f});

    Tensor output = transform(input);

    EXPECT_EQ(output.data().type(), f32);
    EXPECT_FLOAT_EQ(output.data().scalar<float>(), 2.0f);
}

TEST(NormalizeTest, SupportsFloat64Input) {
    Normalize transform({0.5f}, {0.25f});

    const double values[]{1.0, 0.0};

    Tensor input(af::array(1, 2, values));

    Tensor output = transform(input);

    EXPECT_EQ(output.data().type(), f64);

    double actual[2];
    output.data().host(actual);

    EXPECT_NEAR(actual[0], 2.0, kTolerance);
    EXPECT_NEAR(actual[1], -2.0, kTolerance);
}

TEST(NormalizeTest, DoesNotModifyInput) {
    Normalize transform({0.5f}, {0.25f});

    const float values[]{0.0f, 0.5f, 1.0f};

    Tensor input(af::array(1, 3, values));

    Tensor output = transform(input);

    expectValues(input, {0.0f, 0.5f, 1.0f});
    expectValues(output, {-2.0f, 0.0f, 2.0f});
}