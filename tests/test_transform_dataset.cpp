#include "neuro/Tensor.hpp"
#include "neuro/data/Dataset.hpp"
#include "neuro/data/TransformDataset.hpp"
#include "neuro/data/transforms/Compose.hpp"
#include "neuro/data/transforms/Transform.hpp"

#include <gtest/gtest.h>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace neuro;
using namespace neuro::transforms;

namespace {

class TrackingDataset : public Dataset {
public:
    std::size_t size() const override {
        return 3;
    }

    Sample get(std::size_t index) const override {
        ++sample_calls;

        if (index >= size()) {
            throw std::out_of_range("Invalid sample index");
        }

        return {Tensor({static_cast<float>(index + 1)}), Tensor({static_cast<float>(index)})};
    }

    std::pair<Tensor, Tensor> getBatch(std::span<const std::size_t> indices) const override {
        ++batch_calls;

        std::vector<float> inputs;
        std::vector<float> targets;

        for (std::size_t index : indices) {
            if (index >= size()) {
                throw std::out_of_range("Invalid sample index");
            }

            inputs.push_back(static_cast<float>(index + 1));
            targets.push_back(static_cast<float>(index));
        }

        const auto n = static_cast<dim_t>(indices.size());

        return {Tensor(af::array(n, 1, inputs.data(), afHost)),
                Tensor(af::array(n, 1, targets.data(), afHost))};
    }

    mutable int sample_calls = 0;
    mutable int batch_calls = 0;
};

class TrackingTransform : public Transform {
public:
    TrackingTransform(double offset, bool batch_support)
        : offset_(offset), batch_support_(batch_support) {}

    Tensor forward(const Tensor& input) const override {
        ++sample_calls;
        return input + offset_;
    }

    bool supportsBatch() const noexcept override {
        return batch_support_;
    }

    Tensor forwardBatch(const Tensor& input) const override {
        if (!batch_support_) {
            throw std::runtime_error("Batch processing unsupported");
        }

        ++batch_calls;
        return input + offset_;
    }

    mutable int sample_calls = 0;
    mutable int batch_calls = 0;

private:
    double offset_;
    bool batch_support_;
};

void expectValues(const Tensor& tensor, const std::vector<float>& expected) {
    ASSERT_EQ(tensor.data().elements(), expected.size());

    std::vector<float> actual(expected.size());
    tensor.data().host(actual.data());

    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(actual[i], expected[i]) << "Index: " << i;
    }
}

} // namespace

TEST(TransformDatasetTest, DelegatesSize) {
    TrackingDataset dataset;
    TransformDataset transformed(dataset);

    EXPECT_EQ(transformed.size(), dataset.size());
}

TEST(TransformDatasetTest, ReturnsUnchangedSampleWithoutTransforms) {
    TrackingDataset dataset;
    TransformDataset transformed(dataset);

    auto sample = transformed.get(1);

    expectValues(sample.input, {2.0f});
    expectValues(sample.target, {1.0f});
}

TEST(TransformDatasetTest, AppliesInputTransform) {
    TrackingDataset dataset;

    TransformDataset transformed(dataset, std::make_unique<TrackingTransform>(10.0, false));

    auto sample = transformed.get(1);

    expectValues(sample.input, {12.0f});
    expectValues(sample.target, {1.0f});
}

TEST(TransformDatasetTest, AppliesTargetTransform) {
    TrackingDataset dataset;

    TransformDataset transformed(dataset, nullptr,
                                 std::make_unique<TrackingTransform>(20.0, false));

    auto sample = transformed.get(1);

    expectValues(sample.input, {2.0f});
    expectValues(sample.target, {21.0f});
}

TEST(TransformDatasetTest, AppliesBothTransforms) {
    TrackingDataset dataset;

    TransformDataset transformed(dataset, std::make_unique<TrackingTransform>(10.0, false),
                                 std::make_unique<TrackingTransform>(20.0, false));

    auto sample = transformed.get(1);

    expectValues(sample.input, {12.0f});
    expectValues(sample.target, {21.0f});
}

TEST(TransformDatasetTest, PropagatesInvalidIndex) {
    TrackingDataset dataset;
    TransformDataset transformed(dataset);

    EXPECT_THROW(transformed.get(3), std::out_of_range);
}

TEST(TransformDatasetTest, UsesOptimizedBatchWithoutTransforms) {
    TrackingDataset dataset;
    TransformDataset transformed(dataset);

    const std::vector<std::size_t> indices{2, 0, 1};

    auto [inputs, targets] = transformed.getBatch(indices);

    expectValues(inputs, {3.0f, 1.0f, 2.0f});
    expectValues(targets, {2.0f, 0.0f, 1.0f});

    EXPECT_EQ(dataset.batch_calls, 1);
    EXPECT_EQ(dataset.sample_calls, 0);
}

TEST(TransformDatasetTest, UsesOptimizedBatchWithInputTransform) {
    TrackingDataset dataset;

    auto transform = std::make_unique<TrackingTransform>(10.0, true);
    auto* observer = transform.get();

    TransformDataset transformed(dataset, std::move(transform));

    const std::vector<std::size_t> indices{0, 2};

    auto [inputs, targets] = transformed.getBatch(indices);

    expectValues(inputs, {11.0f, 13.0f});
    expectValues(targets, {0.0f, 2.0f});

    EXPECT_EQ(dataset.batch_calls, 1);
    EXPECT_EQ(dataset.sample_calls, 0);

    EXPECT_EQ(observer->batch_calls, 1);
    EXPECT_EQ(observer->sample_calls, 0);
}

TEST(TransformDatasetTest, UsesOptimizedBatchWithTargetTransform) {
    TrackingDataset dataset;

    auto transform = std::make_unique<TrackingTransform>(20.0, true);
    auto* observer = transform.get();

    TransformDataset transformed(dataset, nullptr, std::move(transform));

    const std::vector<std::size_t> indices{0, 2};

    auto [inputs, targets] = transformed.getBatch(indices);

    expectValues(inputs, {1.0f, 3.0f});
    expectValues(targets, {20.0f, 22.0f});

    EXPECT_EQ(dataset.batch_calls, 1);
    EXPECT_EQ(dataset.sample_calls, 0);

    EXPECT_EQ(observer->batch_calls, 1);
    EXPECT_EQ(observer->sample_calls, 0);
}

TEST(TransformDatasetTest, UsesOptimizedBatchWithBothTransforms) {
    TrackingDataset dataset;

    auto input = std::make_unique<TrackingTransform>(10.0, true);
    auto target = std::make_unique<TrackingTransform>(20.0, true);

    auto* input_observer = input.get();
    auto* target_observer = target.get();

    TransformDataset transformed(dataset, std::move(input), std::move(target));

    const std::vector<std::size_t> indices{0, 1, 2};

    auto [inputs, targets] = transformed.getBatch(indices);

    expectValues(inputs, {11.0f, 12.0f, 13.0f});
    expectValues(targets, {20.0f, 21.0f, 22.0f});

    EXPECT_EQ(dataset.batch_calls, 1);
    EXPECT_EQ(dataset.sample_calls, 0);

    EXPECT_EQ(input_observer->batch_calls, 1);
    EXPECT_EQ(target_observer->batch_calls, 1);

    EXPECT_EQ(input_observer->sample_calls, 0);
    EXPECT_EQ(target_observer->sample_calls, 0);
}

TEST(TransformDatasetTest, FallsBackForUnsupportedInputTransform) {
    TrackingDataset dataset;

    auto input = std::make_unique<TrackingTransform>(10.0, false);
    auto target = std::make_unique<TrackingTransform>(20.0, true);

    auto* input_observer = input.get();
    auto* target_observer = target.get();

    TransformDataset transformed(dataset, std::move(input), std::move(target));

    const std::vector<std::size_t> indices{0, 1, 2};

    auto [inputs, targets] = transformed.getBatch(indices);

    expectValues(inputs, {11.0f, 12.0f, 13.0f});
    expectValues(targets, {20.0f, 21.0f, 22.0f});

    EXPECT_EQ(dataset.batch_calls, 0);
    EXPECT_EQ(dataset.sample_calls, 3);

    EXPECT_EQ(input_observer->sample_calls, 3);
    EXPECT_EQ(target_observer->sample_calls, 3);

    EXPECT_EQ(input_observer->batch_calls, 0);
    EXPECT_EQ(target_observer->batch_calls, 0);
}

TEST(TransformDatasetTest, FallsBackForUnsupportedTargetTransform) {
    TrackingDataset dataset;

    auto input = std::make_unique<TrackingTransform>(10.0, true);
    auto target = std::make_unique<TrackingTransform>(20.0, false);

    auto* input_observer = input.get();
    auto* target_observer = target.get();

    TransformDataset transformed(dataset, std::move(input), std::move(target));

    const std::vector<std::size_t> indices{0, 2};

    auto [inputs, targets] = transformed.getBatch(indices);

    expectValues(inputs, {11.0f, 13.0f});
    expectValues(targets, {20.0f, 22.0f});

    EXPECT_EQ(dataset.batch_calls, 0);
    EXPECT_EQ(dataset.sample_calls, 2);

    EXPECT_EQ(input_observer->sample_calls, 2);
    EXPECT_EQ(target_observer->sample_calls, 2);

    EXPECT_EQ(input_observer->batch_calls, 0);
    EXPECT_EQ(target_observer->batch_calls, 0);
}

TEST(TransformDatasetTest, SupportsComposePipeline) {
    TrackingDataset dataset;

    auto pipeline =
        std::make_unique<Compose>(TrackingTransform(10.0, true), TrackingTransform(5.0, true));

    TransformDataset transformed(dataset, std::move(pipeline));

    const std::vector<std::size_t> indices{0, 2};

    auto [inputs, targets] = transformed.getBatch(indices);

    expectValues(inputs, {16.0f, 18.0f});
    expectValues(targets, {0.0f, 2.0f});

    EXPECT_EQ(dataset.batch_calls, 1);
    EXPECT_EQ(dataset.sample_calls, 0);
}

TEST(TransformDatasetTest, FallsBackForMixedComposePipeline) {
    TrackingDataset dataset;

    auto pipeline =
        std::make_unique<Compose>(TrackingTransform(10.0, false), TrackingTransform(5.0, true));

    TransformDataset transformed(dataset, std::move(pipeline));

    const std::vector<std::size_t> indices{2, 0, 2};

    auto [inputs, targets] = transformed.getBatch(indices);

    expectValues(inputs, {18.0f, 16.0f, 18.0f});
    expectValues(targets, {2.0f, 0.0f, 2.0f});

    EXPECT_EQ(dataset.batch_calls, 0);
    EXPECT_EQ(dataset.sample_calls, 3);
}

TEST(TransformDatasetTest, CanReuseUnderlyingDataset) {
    TrackingDataset dataset;

    TransformDataset first(dataset, std::make_unique<TrackingTransform>(10.0, true));

    TransformDataset second(dataset, std::make_unique<TrackingTransform>(20.0, true));

    expectValues(first.get(0).input, {11.0f});
    expectValues(second.get(0).input, {21.0f});
    expectValues(dataset.get(0).input, {1.0f});
}