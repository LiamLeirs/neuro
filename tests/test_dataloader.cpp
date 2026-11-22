
#include "TestUtils.hpp"
#include "neuro/Tensor.hpp"
#include "neuro/data/DataLoader.hpp"

#include <algorithm>
#include <arrayfire.h>
#include <gtest/gtest.h>
#include <initializer_list>
#include <set>
#include <stdexcept>
#include <vector>

using namespace neuro;
using neuro::testing::expectArrayNear;

using Matrix = std::initializer_list<std::initializer_list<float>>;

// ==================== Constructor Validation ====================

TEST(DataLoader, InvalidBatchSize) {
    Tensor input(Matrix{{1, 2}, {3, 4}});
    Tensor target(Matrix{{5}, {6}});

    EXPECT_THROW(DataLoader(input, target, 0), std::invalid_argument);
    EXPECT_THROW(DataLoader(input, target, -1), std::invalid_argument);
}

TEST(DataLoader, MismatchedSampleCounts) {
    Tensor input(Matrix{{1, 2}, {3, 4}});
    Tensor target(Matrix{{5}, {6}, {7}});

    EXPECT_THROW(DataLoader(input, target, 2), std::invalid_argument);
}

// ==================== Batch Iteration ====================

TEST(DataLoader, ExactBatchDivision) {
    Tensor input(Matrix{{1, 2}, {3, 4}, {5, 6}, {7, 8}});

    Tensor target(Matrix{{10}, {20}, {30}, {40}});

    DataLoader loader(input, target, 2);

    int batches = 0;

    for (auto [x, y] : loader) {
        EXPECT_EQ(x.shape()[0], 2);
        EXPECT_EQ(x.shape()[1], 2);
        EXPECT_EQ(y.shape()[0], 2);
        EXPECT_EQ(y.shape()[1], 1);
        ++batches;
    }

    EXPECT_EQ(batches, 2);
}

TEST(DataLoader, IncompleteFinalBatch) {
    Tensor input(Matrix{{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}, {11, 12}, {13, 14}, {15, 16}});

    Tensor target(Matrix{{10}, {20}, {30}, {40}, {50}, {60}, {70}, {80}});

    DataLoader loader(input, target, 3);

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
    Tensor input(Matrix{{1, 2}, {3, 4}});
    Tensor target(Matrix{{10}, {20}});

    DataLoader loader(input, target, 10);

    int batches = 0;

    for (auto [x, y] : loader) {
        expectArrayNear(x.data(), af::dim4(2, 2), {1, 3, 2, 4});

        expectArrayNear(y.data(), af::dim4(2, 1), {10, 20});

        ++batches;
    }

    EXPECT_EQ(batches, 1);
}

TEST(DataLoader, BatchSizeOne) {
    Tensor input(Matrix{{1, 2}, {3, 4}, {5, 6}});
    Tensor target(Matrix{{10}, {20}, {30}});

    DataLoader loader(input, target, 1);

    int batches = 0;

    for (auto [x, y] : loader) {
        EXPECT_EQ(x.shape()[0], 1);
        EXPECT_EQ(y.shape()[0], 1);
        ++batches;
    }

    EXPECT_EQ(batches, 3);
}

// ==================== Data Integrity ====================

TEST(DataLoader, CorrectBatchContents) {
    Tensor input(Matrix{{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}});

    Tensor target(Matrix{{10}, {20}, {30}, {40}, {50}});

    DataLoader loader(input, target, 2);

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
    Tensor input(Matrix{{1, 2}, {3, 4}, {5, 6}});
    Tensor target(Matrix{{10}, {20}, {30}});

    DataLoader loader(input, target, 2);

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
    Tensor input(Matrix{{1}, {2}, {3}, {4}});
    Tensor target(Matrix{{10}, {20}, {30}, {40}});

    DataLoader loader(input, target, 2);

    auto it = loader.begin();
    auto end = loader.end();

    EXPECT_TRUE(it != end);

    ++it;
    EXPECT_TRUE(it != end);

    ++it;
    EXPECT_FALSE(it != end);
}

TEST(DataLoaderIterator, DifferentLoaders) {
    Tensor input(Matrix{{1}, {2}});
    Tensor target(Matrix{{10}, {20}});

    DataLoader first(input, target, 1);
    DataLoader second(input, target, 1);

    EXPECT_TRUE(first.begin() != second.begin());
    EXPECT_TRUE(first.end() != second.end());
}

// ==================== Shuffled Batching ====================

TEST(DataLoaderShuffle, EverySampleAppearsExactlyOnce) {
    Tensor input(Matrix{{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}});
    Tensor target(Matrix{{0}, {10}, {20}, {30}, {40}, {50}, {60}, {70}});

    DataLoader loader(input, target, 3, true);

    std::vector<int> seen;

    for (auto [x, y] : loader) {
        std::vector<float> values(x.data().elements());
        x.data().host(values.data());

        for (float value : values)
            seen.push_back(static_cast<int>(value));
    }

    std::sort(seen.begin(), seen.end());

    EXPECT_EQ(seen, (std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7}));
}

TEST(DataLoaderShuffle, InputTargetAlignment) {
    Tensor input(Matrix{{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}});
    Tensor target(Matrix{{0}, {10}, {20}, {30}, {40}, {50}, {60}, {70}});

    DataLoader loader(input, target, 3, true);

    for (auto [x, y] : loader) {
        std::vector<float> inputs(x.data().elements());
        std::vector<float> targets(y.data().elements());

        x.data().host(inputs.data());
        y.data().host(targets.data());

        ASSERT_EQ(inputs.size(), targets.size());

        for (size_t i = 0; i < inputs.size(); ++i)
            EXPECT_FLOAT_EQ(targets[i], inputs[i] * 10.0f);
    }
}

TEST(DataLoaderShuffle, IncompleteFinalBatch) {
    Tensor input(Matrix{{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}});
    Tensor target(Matrix{{0}, {10}, {20}, {30}, {40}, {50}, {60}, {70}});

    DataLoader loader(input, target, 3, true);

    std::vector<int> batchSizes;

    for (auto [x, y] : loader) {
        batchSizes.push_back(static_cast<int>(x.shape()[0]));
        EXPECT_EQ(x.shape()[0], y.shape()[0]);
    }

    EXPECT_EQ(batchSizes, (std::vector<int>{3, 3, 2}));
}

TEST(DataLoaderShuffle, RepeatedEpochsPreserveAllSamples) {
    Tensor input(Matrix{{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}});
    Tensor target(Matrix{{0}, {10}, {20}, {30}, {40}, {50}, {60}, {70}});

    DataLoader loader(input, target, 3, true);

    for (int epoch = 0; epoch < 5; ++epoch) {
        std::vector<int> seen;

        for (auto [x, y] : loader) {
            std::vector<float> values(x.data().elements());
            x.data().host(values.data());

            for (float value : values)
                seen.push_back(static_cast<int>(value));
        }

        std::sort(seen.begin(), seen.end());

        EXPECT_EQ(seen, (std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7}));
    }
}

TEST(DataLoaderShuffle, SameSeedProducesSameOrder) {
    Tensor input(Matrix{{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}});
    Tensor target(Matrix{{0}, {10}, {20}, {30}, {40}, {50}, {60}, {70}});

    DataLoader first(input, target, 3, true, 123);
    DataLoader second(input, target, 3, true, 123);

    for (int epoch = 0; epoch < 3; ++epoch) {
        std::vector<float> firstOrder;
        std::vector<float> secondOrder;

        for (auto [x, y] : first) {
            std::vector<float> values(x.data().elements());
            x.data().host(values.data());

            firstOrder.insert(firstOrder.end(), values.begin(), values.end());
        }

        for (auto [x, y] : second) {
            std::vector<float> values(x.data().elements());
            x.data().host(values.data());

            secondOrder.insert(secondOrder.end(), values.begin(), values.end());
        }

        EXPECT_EQ(firstOrder, secondOrder);
    }
}

TEST(DataLoaderShuffle, ShuffleDisabledPreservesOrder) {
    Tensor input(Matrix{{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}});
    Tensor target(Matrix{{0}, {10}, {20}, {30}, {40}, {50}, {60}, {70}});

    DataLoader loader(input, target, 3, false);

    for (int epoch = 0; epoch < 3; ++epoch) {
        std::vector<float> seen;

        for (auto [x, y] : loader) {
            std::vector<float> values(x.data().elements());
            x.data().host(values.data());

            seen.insert(seen.end(), values.begin(), values.end());
        }

        EXPECT_EQ(seen, (std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7}));
    }
}

TEST(DataLoaderShuffle, SingleSample) {
    Tensor input(Matrix{{42}});
    Tensor target(Matrix{{420}});

    DataLoader loader(input, target, 3, true);

    int batches = 0;

    for (auto [x, y] : loader) {
        expectArrayNear(x.data(), af::dim4(1, 1), {42});
        expectArrayNear(y.data(), af::dim4(1, 1), {420});
        ++batches;
    }

    EXPECT_EQ(batches, 1);
}
