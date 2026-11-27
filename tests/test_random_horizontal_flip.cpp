#include "neuro/Tensor.hpp"
#include "neuro/data/transforms/RandomHorizontalFlip.hpp"

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace neuro;
using namespace neuro::transforms;

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

TEST(RandomHorizontalFlipTest, RejectsInvalidProbability) {
    const float infinity_value = std::numeric_limits<float>::infinity();
    const float nan_value = std::numeric_limits<float>::quiet_NaN();

    EXPECT_THROW(RandomHorizontalFlip{-0.1f}, std::invalid_argument);
    EXPECT_THROW(RandomHorizontalFlip{1.1f}, std::invalid_argument);
    EXPECT_THROW(RandomHorizontalFlip{infinity_value}, std::invalid_argument);
    EXPECT_THROW(RandomHorizontalFlip{nan_value}, std::invalid_argument);
}

TEST(RandomHorizontalFlipTest, ZeroProbabilityPreservesImage) {
    RandomHorizontalFlip transform(0.0f);

    const float values[]{1, 2, 3, 4, 5, 6};
    Tensor input(af::array(1, 2, 3, values));

    Tensor output = transform(input);

    EXPECT_EQ(output.shape(), input.shape());
    expectValues(output.data(), {1, 2, 3, 4, 5, 6});
}

TEST(RandomHorizontalFlipTest, OneProbabilityFlipsImage) {
    RandomHorizontalFlip transform(1.0f);

    const float values[]{1, 2, 3, 4, 5, 6};
    Tensor input(af::array(1, 2, 3, values));

    Tensor output = transform(input);

    EXPECT_EQ(output.shape(), input.shape());
    expectValues(output.data(), {5, 6, 3, 4, 1, 2});
}

TEST(RandomHorizontalFlipTest, PreservesRGBChannelOrder) {
    RandomHorizontalFlip transform(1.0f);

    const float values[]{1, 2, 3, 4, 5, 6, 7, 8, 9};

    Tensor input(af::array(3, 1, 3, values));

    expectValues(transform(input).data(), {7, 8, 9, 4, 5, 6, 1, 2, 3});
}

TEST(RandomHorizontalFlipTest, SupportsBatching) {
    RandomHorizontalFlip transform;

    EXPECT_TRUE(transform.supportsBatch());
}

TEST(RandomHorizontalFlipTest, ZeroProbabilityPreservesBatch) {
    RandomHorizontalFlip transform(0.0f);

    const float values[]{1, 2, 3, 4, 5, 6, 7, 8};
    Tensor input(af::array(2, 1, 2, 2, values));

    Tensor output = transform.forwardBatch(input);

    EXPECT_EQ(output.shape(), input.shape());
    expectValues(output.data(), {1, 2, 3, 4, 5, 6, 7, 8});
}

TEST(RandomHorizontalFlipTest, OneProbabilityFlipsEntireBatch) {
    RandomHorizontalFlip transform(1.0f);

    const float values[]{1, 2, 3, 4, 5, 6, 7, 8};
    Tensor input(af::array(2, 1, 2, 2, values));

    Tensor output = transform.forwardBatch(input);

    EXPECT_EQ(output.shape(), input.shape());
    expectValues(output.data(), {5, 6, 7, 8, 1, 2, 3, 4});
}

TEST(RandomHorizontalFlipTest, MakesIndependentBatchDecisions) {
    constexpr int n = 256;

    RandomHorizontalFlip transform(0.5f);

    std::vector<float> values(n * 2);

    for (int i = 0; i < n; ++i) {
        values[i] = 1.0f;
        values[n + i] = 2.0f;
    }

    Tensor input(af::array(n, 1, 1, 2, values.data()));

    Tensor output = transform.forwardBatch(input);

    ASSERT_EQ(output.shape(), input.shape());

    std::vector<float> actual(values.size());
    output.data().host(actual.data());

    int flipped = 0;

    for (int i = 0; i < n; ++i) {
        const float left = actual[i];
        const float right = actual[n + i];

        const bool unchanged = left == 1.0f && right == 2.0f;
        const bool reversed = left == 2.0f && right == 1.0f;

        EXPECT_TRUE(unchanged || reversed);

        if (reversed) {
            ++flipped;
        }
    }

    EXPECT_GT(flipped, 0);
    EXPECT_LT(flipped, n);
}

TEST(RandomHorizontalFlipTest, RespectsConfiguredProbability) {
    constexpr int n = 4096;
    constexpr float p = 0.2f;

    RandomHorizontalFlip transform(p);

    std::vector<float> values(n * 2);

    for (int i = 0; i < n; ++i) {
        values[i] = 1.0f;
        values[n + i] = 2.0f;
    }

    Tensor input(af::array(n, 1, 1, 2, values.data()));

    Tensor output = transform.forwardBatch(input);

    std::vector<float> actual(values.size());
    output.data().host(actual.data());

    int flipped = 0;

    for (int i = 0; i < n; ++i) {
        if (actual[i] == 2.0f) {
            ++flipped;
        }
    }

    const double observed = static_cast<double>(flipped) / n;

    EXPECT_NEAR(observed, p, 0.05);
}

TEST(RandomHorizontalFlipTest, FlipsWholeImagesNotIndividualPixels) {
    constexpr int n = 32;
    constexpr int c = 3;
    constexpr int h = 2;
    constexpr int w = 4;

    RandomHorizontalFlip transform(0.5f);

    std::vector<float> values(n * c * h * w);

    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(i);
    }

    Tensor input(af::array(n, c, h, w, values.data()));

    Tensor output = transform.forwardBatch(input);

    ASSERT_EQ(output.shape(), input.shape());

    std::vector<float> actual(values.size());
    output.data().host(actual.data());

    for (int b = 0; b < n; ++b) {
        const bool flipped = actual[b] != values[b];

        for (int x = 0; x < w; ++x) {
            for (int y = 0; y < h; ++y) {
                for (int channel = 0; channel < c; ++channel) {
                    const int source_x = flipped ? w - 1 - x : x;

                    const int output_index = b + n * (channel + c * (y + h * x));

                    const int source_index = b + n * (channel + c * (y + h * source_x));

                    EXPECT_FLOAT_EQ(actual[output_index], values[source_index]);
                }
            }
        }
    }
}

TEST(RandomHorizontalFlipTest, PreservesIntegerType) {
    RandomHorizontalFlip transform(1.0f);

    const unsigned char values[]{1, 2, 3};
    Tensor input(af::array(1, 1, 3, values));

    Tensor output = transform(input);

    EXPECT_EQ(output.data().type(), u8);

    unsigned char actual[3];
    output.data().host(actual);

    EXPECT_EQ(actual[0], 3);
    EXPECT_EQ(actual[1], 2);
    EXPECT_EQ(actual[2], 1);
}

TEST(RandomHorizontalFlipTest, DoesNotModifyInput) {
    RandomHorizontalFlip transform(1.0f);

    const float values[]{1, 2, 3, 4};
    Tensor input(af::array(1, 2, 2, values));

    Tensor output = transform(input);

    expectValues(input.data(), {1, 2, 3, 4});
    expectValues(output.data(), {3, 4, 1, 2});
}

TEST(RandomHorizontalFlipTest, RejectsEmptyInput) {
    RandomHorizontalFlip transform;
    Tensor input;

    EXPECT_THROW(transform(input), std::invalid_argument);

    EXPECT_THROW(transform.forwardBatch(input), std::invalid_argument);
}

TEST(RandomHorizontalFlipTest, BackwardThroughBatchAugmentation) {
    constexpr int n = 32;

    RandomHorizontalFlip transform(0.5f);

    std::vector<float> values(n * 2);
    std::vector<float> weights(n * 2);

    for (int i = 0; i < n; ++i) {
        values[i] = 1.0f;
        values[n + i] = 2.0f;

        weights[i] = 3.0f;
        weights[n + i] = 7.0f;
    }

    Tensor input(af::array(n, 1, 1, 2, values.data()), true);

    Tensor weight(af::array(n, 1, 1, 2, weights.data()));

    Tensor output = transform.forwardBatch(input);

    std::vector<float> actual(values.size());
    output.data().host(actual.data());

    Tensor loss = (output * weight).sum();
    loss.backward();

    std::vector<float> expected_grad(values.size());

    for (int i = 0; i < n; ++i) {
        const bool flipped = actual[i] == 2.0f;

        expected_grad[i] = flipped ? 7.0f : 3.0f;
        expected_grad[n + i] = flipped ? 3.0f : 7.0f;
    }

    expectValues(input.grad(), expected_grad);
}