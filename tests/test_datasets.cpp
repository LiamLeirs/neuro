
#include "TestUtils.hpp"
#include "neuro/data/TensorDataset.hpp"

#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

using namespace neuro;
using neuro::testing::expectArrayNear;

TEST(TensorDatasetTest, Construction) {
    std::vector<Tensor> inputs = {Tensor({1.0f, 2.0f}), Tensor({3.0f, 4.0f})};

    std::vector<Tensor> targets = {Tensor({0.0f}), Tensor({1.0f})};

    TensorDataset dataset(inputs, targets);

    EXPECT_EQ(dataset.size(), 2);

    TensorDataset empty({}, {});
    EXPECT_EQ(empty.size(), 0);
}

TEST(TensorDatasetTest, Retrieval) {
    std::vector<Tensor> inputs = {Tensor({1.0f, 2.0f}), Tensor({3.0f, 4.0f})};

    std::vector<Tensor> targets = {Tensor({0.0f}), Tensor({1.0f})};

    TensorDataset dataset(inputs, targets);

    auto sample = dataset.get(1);

    expectArrayNear(sample.input.data(), af::dim4(2), {3.0f, 4.0f});

    expectArrayNear(sample.target.data(), af::dim4(1), {1.0f});
}

TEST(TensorDatasetTest, Validation) {
    std::vector<Tensor> inputs = {Tensor({1.0f}), Tensor({2.0f})};

    std::vector<Tensor> targets = {Tensor({0.0f})};

    EXPECT_THROW(TensorDataset(inputs, targets), std::invalid_argument);

    TensorDataset dataset({Tensor({1.0f})}, {Tensor({0.0f})});

    EXPECT_THROW(dataset.get(1), std::out_of_range);
}
