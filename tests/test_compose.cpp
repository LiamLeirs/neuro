#include "neuro/Tensor.hpp"
#include "neuro/data/transforms/Compose.hpp"
#include "neuro/data/transforms/Transform.hpp"

#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>

using namespace neuro;
using namespace neuro::transforms;

namespace {

class AddConstant : public Transform {
public:
    explicit AddConstant(double value) : value_(value) {}

    Tensor forward(const Tensor& input) const override {
        return input + value_;
    }

private:
    double value_;
};

class MultiplyConstant : public Transform {
public:
    explicit MultiplyConstant(double value) : value_(value) {}

    Tensor forward(const Tensor& input) const override {
        return input * value_;
    }

private:
    double value_;
};

} // namespace

TEST(ComposeTest, EmptyPipelineReturnsInput) {
    Compose pipeline;
    Tensor input({2.0f, 3.0f, 4.0f});

    Tensor output = pipeline(input);

    EXPECT_EQ(output.shape(), input.shape());

    float values[3];
    output.data().host(values);

    EXPECT_FLOAT_EQ(values[0], 2.0f);
    EXPECT_FLOAT_EQ(values[1], 3.0f);
    EXPECT_FLOAT_EQ(values[2], 4.0f);
}

TEST(ComposeTest, AppliesSingleTransform) {
    Compose pipeline(AddConstant(3.0));
    Tensor input({2.0f});

    Tensor output = pipeline(input);

    EXPECT_FLOAT_EQ(output.data().scalar<float>(), 5.0f);
}

TEST(ComposeTest, AppliesTransformsInInsertionOrder) {
    Compose pipeline(AddConstant(3.0), MultiplyConstant(4.0));

    Tensor input({2.0f});

    Tensor output = pipeline(input);

    EXPECT_FLOAT_EQ(output.data().scalar<float>(), 20.0f);
}

TEST(ComposeTest, SupportsAddingTransforms) {
    Compose pipeline(AddConstant(3.0));

    pipeline.add(std::make_unique<MultiplyConstant>(4.0));

    Tensor input({2.0f});

    EXPECT_FLOAT_EQ(pipeline(input).data().scalar<float>(), 20.0f);
}

TEST(ComposeTest, RejectsNullTransform) {
    Compose pipeline;

    EXPECT_THROW(pipeline.add(nullptr), std::invalid_argument);
}

TEST(ComposeTest, SupportsNestedPipelines) {
    Compose pipeline(AddConstant(3.0));

    auto nested = std::make_unique<Compose>(MultiplyConstant(4.0), AddConstant(1.0));

    pipeline.add(std::move(nested));

    Tensor input({2.0f});

    EXPECT_FLOAT_EQ(pipeline(input).data().scalar<float>(), 21.0f);
}

TEST(ComposeTest, DoesNotModifyInput) {
    Compose pipeline(AddConstant(3.0), MultiplyConstant(4.0));

    Tensor input({2.0f});

    Tensor output = pipeline(input);

    EXPECT_FLOAT_EQ(input.data().scalar<float>(), 2.0f);
    EXPECT_FLOAT_EQ(output.data().scalar<float>(), 20.0f);
}

TEST(ComposeTest, CanBeReusedWithDifferentInputs) {
    Compose pipeline(AddConstant(3.0), MultiplyConstant(4.0));

    Tensor first({2.0f});
    Tensor second({5.0f});

    EXPECT_FLOAT_EQ(pipeline(first).data().scalar<float>(), 20.0f);
    EXPECT_FLOAT_EQ(pipeline(second).data().scalar<float>(), 32.0f);
}

namespace {

class BatchAddConstant : public Transform {
public:
    explicit BatchAddConstant(double value) : value_(value) {}

    Tensor forward(const Tensor& input) const override {
        return input + value_;
    }

    bool supportsBatch() const noexcept override {
        return true;
    }

    Tensor forwardBatch(const Tensor& input) const override {
        ++batch_calls;
        return input + value_;
    }

    mutable int batch_calls = 0;

private:
    double value_;
};

class BatchMultiplyConstant : public Transform {
public:
    explicit BatchMultiplyConstant(double value) : value_(value) {}

    Tensor forward(const Tensor& input) const override {
        return input * value_;
    }

    bool supportsBatch() const noexcept override {
        return true;
    }

    Tensor forwardBatch(const Tensor& input) const override {
        return input * value_;
    }

private:
    double value_;
};

} // namespace

TEST(ComposeTest, EmptyPipelineSupportsBatching) {
    Compose pipeline;
    Tensor input({2.0f, 3.0f});

    EXPECT_TRUE(pipeline.supportsBatch());

    Tensor output = pipeline.forwardBatch(input);

    EXPECT_EQ(output.shape(), input.shape());
    EXPECT_FLOAT_EQ(output.data()(0).scalar<float>(), 2.0f);
    EXPECT_FLOAT_EQ(output.data()(1).scalar<float>(), 3.0f);
}

TEST(ComposeTest, SupportsBatchingWhenAllTransformsSupportIt) {
    Compose pipeline(BatchAddConstant(3.0), BatchMultiplyConstant(4.0));

    EXPECT_TRUE(pipeline.supportsBatch());
}

TEST(ComposeTest, RejectsBatchingWithUnsupportedTransform) {
    Compose first(AddConstant(3.0), BatchMultiplyConstant(4.0));

    Compose second(BatchAddConstant(3.0), MultiplyConstant(4.0));

    EXPECT_FALSE(first.supportsBatch());
    EXPECT_FALSE(second.supportsBatch());
}

TEST(ComposeTest, AppliesBatchTransformsInInsertionOrder) {
    Compose pipeline(BatchAddConstant(3.0), BatchMultiplyConstant(4.0));

    Tensor input({2.0f, 5.0f});
    Tensor output = pipeline.forwardBatch(input);

    EXPECT_FLOAT_EQ(output.data()(0).scalar<float>(), 20.0f);
    EXPECT_FLOAT_EQ(output.data()(1).scalar<float>(), 32.0f);
}

TEST(ComposeTest, RejectsUnsupportedBatchBeforeExecutingTransforms) {
    Compose pipeline;

    auto transform = std::make_unique<BatchAddConstant>(3.0);
    auto* observer = transform.get();

    pipeline.add(std::move(transform));
    pipeline.add(std::make_unique<MultiplyConstant>(4.0));

    Tensor input({2.0f});

    EXPECT_THROW(pipeline.forwardBatch(input), std::runtime_error);
    EXPECT_EQ(observer->batch_calls, 0);
}

TEST(ComposeTest, NestedPipelinePropagatesBatchSupport) {
    Compose pipeline(BatchAddConstant(3.0));

    auto nested = std::make_unique<Compose>(BatchMultiplyConstant(4.0), AddConstant(1.0));

    pipeline.add(std::move(nested));

    EXPECT_FALSE(pipeline.supportsBatch());
}