
#include "TestUtils.hpp"
#include "neuro/Tensor.hpp"
#include "neuro/data/DataLoader.hpp"
#include "neuro/data/TensorDataset.hpp"

#include <algorithm>
#include <arrayfire.h>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

using namespace neuro;
using neuro::testing::expectArrayNear;

using Matrix = std::initializer_list<std::initializer_list<float>>;

// ==================== Test Helpers ====================

static TensorDataset makeDataset(const std::vector<std::vector<float>>& inputs,
                                 const std::vector<std::vector<float>>& targets) {
    std::vector<Tensor> inputTensors;
    std::vector<Tensor> targetTensors;

    inputTensors.reserve(inputs.size());
    targetTensors.reserve(targets.size());

    for (const auto& sample : inputs) {
        af::array data(sample.size(), sample.data());
        inputTensors.emplace_back(data);
    }

    for (const auto& sample : targets) {
        af::array data(sample.size(), sample.data());
        targetTensors.emplace_back(data);
    }

    return TensorDataset(inputTensors, targetTensors);
}

static std::vector<float> hostValues(const Tensor& tensor) {
    std::vector<float> values(tensor.data().elements());
    tensor.data().host(values.data());
    return values;
}

// ==================== Constructor Validation ====================

TEST(DataLoader, InvalidBatchSize) {
    auto dataset = makeDataset({{1, 2}, {3, 4}}, {{5}, {6}});

    EXPECT_THROW(DataLoader(dataset, 0), std::invalid_argument);
    EXPECT_THROW(DataLoader(dataset, -1), std::invalid_argument);
}

TEST(DataLoader, MismatchedSampleCounts) {
    EXPECT_THROW(makeDataset({{1, 2}, {3, 4}}, {{5}, {6}, {7}}), std::invalid_argument);
}

// ==================== Batch Iteration ====================

TEST(DataLoader, ExactBatchDivision) {
    auto dataset = makeDataset({{1, 2}, {3, 4}, {5, 6}, {7, 8}}, {{10}, {20}, {30}, {40}});

    DataLoader loader(dataset, 2);

    int batches = 0;

    for (auto [x, y] : loader) {
        EXPECT_EQ(x.shape(), af::dim4(2, 2));
        EXPECT_EQ(y.shape(), af::dim4(2, 1));
        ++batches;
    }

    EXPECT_EQ(batches, 2);
}

TEST(DataLoader, IncompleteFinalBatch) {
    auto dataset =
        makeDataset({{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}, {11, 12}, {13, 14}, {15, 16}},
                    {{10}, {20}, {30}, {40}, {50}, {60}, {70}, {80}});

    DataLoader loader(dataset, 3);

    std::vector<int> batchSizes;

    for (auto [x, y] : loader) {
        batchSizes.push_back(static_cast<int>(x.shape()[0]));

        EXPECT_EQ(x.shape()[0], y.shape()[0]);
        EXPECT_EQ(x.shape()[1], 2);
        EXPECT_EQ(y.shape()[1], 1);
    }

    EXPECT_EQ(batchSizes, (std::vector<int>{3, 3, 2}));
}

TEST(DataLoader, BatchLargerThanDataset) {
    auto dataset = makeDataset({{1, 2}, {3, 4}}, {{10}, {20}});

    DataLoader loader(dataset, 10);

    int batches = 0;

    for (auto [x, y] : loader) {
        expectArrayNear(x.data(), af::dim4(2, 2), {1, 3, 2, 4});
        expectArrayNear(y.data(), af::dim4(2, 1), {10, 20});
        ++batches;
    }

    EXPECT_EQ(batches, 1);
}

TEST(DataLoader, BatchSizeOne) {
    auto dataset = makeDataset({{1, 2}, {3, 4}, {5, 6}}, {{10}, {20}, {30}});

    DataLoader loader(dataset, 1);

    int batches = 0;

    for (auto [x, y] : loader) {
        EXPECT_EQ(x.shape(), af::dim4(1, 2));
        EXPECT_EQ(y.shape(), af::dim4(1, 1));
        ++batches;
    }

    EXPECT_EQ(batches, 3);
}

// ==================== Data Integrity ====================

TEST(DataLoader, CorrectBatchContents) {
    auto dataset =
        makeDataset({{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}}, {{10}, {20}, {30}, {40}, {50}});

    DataLoader loader(dataset, 2);

    int batch = 0;

    for (auto [x, y] : loader) {
        if (batch == 0) {
            expectArrayNear(x.data(), af::dim4(2, 2), {1, 3, 2, 4});
            expectArrayNear(y.data(), af::dim4(2, 1), {10, 20});
        } else if (batch == 1) {
            expectArrayNear(x.data(), af::dim4(2, 2), {5, 7, 6, 8});
            expectArrayNear(y.data(), af::dim4(2, 1), {30, 40});
        } else if (batch == 2) {
            expectArrayNear(x.data(), af::dim4(1, 2), {9, 10});
            expectArrayNear(y.data(), af::dim4(1, 1), {50});
        }

        ++batch;
    }

    EXPECT_EQ(batch, 3);
}

TEST(DataLoader, RepeatedEpochs) {
    auto dataset = makeDataset({{1, 2}, {3, 4}, {5, 6}}, {{10}, {20}, {30}});

    DataLoader loader(dataset, 2);

    for (int epoch = 0; epoch < 3; ++epoch) {
        int samples = 0;

        for (auto [x, y] : loader) {
            samples += static_cast<int>(x.shape()[0]);
            EXPECT_EQ(x.shape()[0], y.shape()[0]);
        }

        EXPECT_EQ(samples, 3);
    }
}

// ==================== Iterator Behavior ====================

TEST(DataLoaderIterator, BeginAndEnd) {
    auto dataset = makeDataset({{1}, {2}, {3}, {4}}, {{10}, {20}, {30}, {40}});

    DataLoader loader(dataset, 2);

    auto it = loader.begin();
    auto end = loader.end();

    EXPECT_TRUE(it != end);

    ++it;
    EXPECT_TRUE(it != end);

    ++it;
    EXPECT_FALSE(it != end);
}

TEST(DataLoaderIterator, DifferentLoaders) {
    auto dataset = makeDataset({{1}, {2}}, {{10}, {20}});

    DataLoader first(dataset, 1);
    DataLoader second(dataset, 1);

    EXPECT_TRUE(first.begin() != second.begin());
    EXPECT_TRUE(first.end() != second.end());
}

// ==================== Shuffled Batching ====================

TEST(DataLoaderShuffle, EverySampleAppearsExactlyOnce) {
    auto dataset = makeDataset({{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}},
                               {{0}, {10}, {20}, {30}, {40}, {50}, {60}, {70}});

    DataLoader loader(dataset, 3, true);

    std::vector<int> seen;

    for (auto [x, y] : loader) {
        for (float value : hostValues(x))
            seen.push_back(static_cast<int>(value));
    }

    std::sort(seen.begin(), seen.end());

    EXPECT_EQ(seen, (std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7}));
}

TEST(DataLoaderShuffle, InputTargetAlignment) {
    auto dataset = makeDataset({{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}},
                               {{0}, {10}, {20}, {30}, {40}, {50}, {60}, {70}});

    DataLoader loader(dataset, 3, true);

    for (auto [x, y] : loader) {
        auto inputs = hostValues(x);
        auto targets = hostValues(y);

        ASSERT_EQ(inputs.size(), targets.size());

        for (std::size_t i = 0; i < inputs.size(); ++i)
            EXPECT_FLOAT_EQ(targets[i], inputs[i] * 10.0f);
    }
}

TEST(DataLoaderShuffle, IncompleteFinalBatch) {
    auto dataset = makeDataset({{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}},
                               {{0}, {10}, {20}, {30}, {40}, {50}, {60}, {70}});

    DataLoader loader(dataset, 3, true);

    std::vector<int> batchSizes;

    for (auto [x, y] : loader) {
        batchSizes.push_back(static_cast<int>(x.shape()[0]));
        EXPECT_EQ(x.shape()[0], y.shape()[0]);
    }

    EXPECT_EQ(batchSizes, (std::vector<int>{3, 3, 2}));
}

TEST(DataLoaderShuffle, RepeatedEpochsPreserveAllSamples) {
    auto dataset = makeDataset({{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}},
                               {{0}, {10}, {20}, {30}, {40}, {50}, {60}, {70}});

    DataLoader loader(dataset, 3, true);

    for (int epoch = 0; epoch < 5; ++epoch) {
        std::vector<int> seen;

        for (auto [x, y] : loader) {
            for (float value : hostValues(x))
                seen.push_back(static_cast<int>(value));
        }

        std::sort(seen.begin(), seen.end());

        EXPECT_EQ(seen, (std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7}));
    }
}

TEST(DataLoaderShuffle, SameSeedProducesSameOrder) {
    auto dataset = makeDataset({{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}},
                               {{0}, {10}, {20}, {30}, {40}, {50}, {60}, {70}});

    DataLoader first(dataset, 3, true, 123);
    DataLoader second(dataset, 3, true, 123);

    for (int epoch = 0; epoch < 3; ++epoch) {
        std::vector<float> firstOrder;
        std::vector<float> secondOrder;

        for (auto [x, y] : first) {
            auto values = hostValues(x);
            firstOrder.insert(firstOrder.end(), values.begin(), values.end());
        }

        for (auto [x, y] : second) {
            auto values = hostValues(x);
            secondOrder.insert(secondOrder.end(), values.begin(), values.end());
        }

        EXPECT_EQ(firstOrder, secondOrder);
    }
}

TEST(DataLoaderShuffle, ShuffleDisabledPreservesOrder) {
    auto dataset = makeDataset({{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}},
                               {{0}, {10}, {20}, {30}, {40}, {50}, {60}, {70}});

    DataLoader loader(dataset, 3, false);

    for (int epoch = 0; epoch < 3; ++epoch) {
        std::vector<float> seen;

        for (auto [x, y] : loader) {
            auto values = hostValues(x);
            seen.insert(seen.end(), values.begin(), values.end());
        }

        EXPECT_EQ(seen, (std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7}));
    }
}

TEST(DataLoaderShuffle, SingleSample) {
    auto dataset = makeDataset({{42}}, {{420}});

    DataLoader loader(dataset, 3, true);

    int batches = 0;

    for (auto [x, y] : loader) {
        expectArrayNear(x.data(), af::dim4(1, 1), {42});
        expectArrayNear(y.data(), af::dim4(1, 1), {420});
        ++batches;
    }

    EXPECT_EQ(batches, 1);
}
