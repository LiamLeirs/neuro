
#include "neuro/GradMode.hpp"
#include "neuro/Tensor.hpp"
#include "neuro/Trainer.hpp"
#include "neuro/data/DataLoader.hpp"
#include "neuro/data/TensorDataset.hpp"
#include "neuro/metrics/Accuracy.hpp"
#include "neuro/metrics/Metric.hpp"
#include "neuro/nn/Linear.hpp"
#include "neuro/nn/loss/MSELoss.hpp"
#include "neuro/optim/SGD.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace neuro;

namespace {

// Counts the number of samples processed.
// Useful for verifying metric lifecycle management.
class SampleCountMetric : public metrics::Metric {
public:
    void update(const Tensor& predictions, const Tensor& targets) override {
        count_ += predictions.shape()[0];
    }

    double compute() const override {
        return static_cast<double>(count_);
    }

    void reset() override {
        count_ = 0;
    }

private:
    std::size_t count_ = 0;
};

// Verifies that metric updates run without gradient tracking.
class GradModeMetric : public metrics::Metric {
public:
    void update(const Tensor& predictions, const Tensor& targets) override {
        if (GradMode::isEnabled()) {
            enabled_count_++;
        }
    }

    double compute() const override {
        return static_cast<double>(enabled_count_);
    }

    void reset() override {
        enabled_count_ = 0;
    }

private:
    std::size_t enabled_count_ = 0;
};

// Tests forwarding constructor arguments through addMetric().
class ConfigurableMetric : public metrics::Metric {
public:
    explicit ConfigurableMetric(double value) : value_(value) {}

    void update(const Tensor&, const Tensor&) override {}

    double compute() const override {
        return value_;
    }

    void reset() override {}

private:
    double value_;
};

class TrainerTest : public ::testing::Test {
protected:
    // Regression dataset: y = 2x + 1.
    TensorDataset dataset{
        std::vector<Tensor>{Tensor{0.0f}, Tensor{1.0f}, Tensor{2.0f}, Tensor{3.0f}, Tensor{4.0f}},
        std::vector<Tensor>{Tensor{1.0f}, Tensor{3.0f}, Tensor{5.0f}, Tensor{7.0f}, Tensor{9.0f}}};

    DataLoader loader{dataset, 2, false};

    nn::Linear model{1, 1};
    nn::MSELoss criterion;
    SGD optim{model.parameters(), 0.01};

    Trainer trainer{model, optim, criterion, TrainerConfig{.verbose = false}};

    // Copies parameter values to CPU memory.
    std::vector<std::vector<float>> snapshotParameters() {
        std::vector<std::vector<float>> snapshots;

        for (const auto& param : model.parameters()) {
            std::vector<float> values(param.data().elements());
            param.data().as(f32).host(values.data());
            snapshots.push_back(std::move(values));
        }

        return snapshots;
    }

    // Copies gradient values to CPU memory.
    std::vector<std::vector<float>> snapshotGradients() {
        std::vector<std::vector<float>> snapshots;

        for (const auto& param : model.parameters()) {
            std::vector<float> values(param.grad().elements());
            param.grad().as(f32).host(values.data());
            snapshots.push_back(std::move(values));
        }

        return snapshots;
    }
};

} // namespace

// ============================================================
// Constructor
// ============================================================

TEST_F(TrainerTest, AcceptsValidCriterion) {
    EXPECT_NO_THROW(Trainer(model, optim, criterion));
}

TEST_F(TrainerTest, RejectsEmptyCriterion) {
    LossFunction empty;

    EXPECT_THROW(Trainer(model, optim, empty), std::invalid_argument);
}

TEST_F(TrainerTest, AcceptsLambdaCriterion) {
    LossFunction loss = [](const Tensor& predictions, const Tensor& targets) {
        return (predictions - targets).abs().mean();
    };

    EXPECT_NO_THROW(Trainer(model, optim, loss));
}

// ============================================================
// Metric registration
// ============================================================

TEST_F(TrainerTest, RegistersMetricThroughUniquePointer) {
    trainer.addMetric("samples", std::make_unique<SampleCountMetric>());

    auto results = trainer.evaluate(loader);

    EXPECT_TRUE(results.contains("samples"));
    EXPECT_DOUBLE_EQ(results.at("samples"), 5.0);
}

TEST_F(TrainerTest, RegistersMetricThroughTemplate) {
    trainer.addMetric<SampleCountMetric>("samples");

    auto results = trainer.evaluate(loader);

    EXPECT_DOUBLE_EQ(results.at("samples"), 5.0);
}

TEST_F(TrainerTest, RegistersMultipleMetrics) {
    trainer.addMetric<SampleCountMetric>("samples");
    trainer.addMetric<GradModeMetric>("grad_mode");

    auto results = trainer.evaluate(loader);

    EXPECT_EQ(results.size(), 3u);
    EXPECT_TRUE(results.contains("loss"));
    EXPECT_TRUE(results.contains("samples"));
    EXPECT_TRUE(results.contains("grad_mode"));
}

TEST_F(TrainerTest, RejectsEmptyMetricName) {
    EXPECT_THROW(trainer.addMetric<SampleCountMetric>(""), std::invalid_argument);
}

TEST_F(TrainerTest, RejectsNullMetric) {
    EXPECT_THROW(trainer.addMetric("invalid", nullptr), std::invalid_argument);
}

TEST_F(TrainerTest, RejectsDuplicateMetricName) {
    trainer.addMetric<SampleCountMetric>("samples");

    EXPECT_THROW(trainer.addMetric<SampleCountMetric>("samples"), std::invalid_argument);
}

TEST_F(TrainerTest, RejectsReservedLossName) {
    EXPECT_THROW(trainer.addMetric<SampleCountMetric>("loss"), std::invalid_argument);
}

TEST_F(TrainerTest, FailedRegistrationPreservesExistingMetric) {
    trainer.addMetric<ConfigurableMetric>("custom", 42.0);

    EXPECT_THROW(trainer.addMetric<ConfigurableMetric>("custom", 99.0), std::invalid_argument);

    auto results = trainer.evaluate(loader);

    EXPECT_DOUBLE_EQ(results.at("custom"), 42.0);
}

TEST_F(TrainerTest, ForwardsMetricConstructorArguments) {
    trainer.addMetric<ConfigurableMetric>("custom", 42.5);

    auto results = trainer.evaluate(loader);

    EXPECT_DOUBLE_EQ(results.at("custom"), 42.5);
}

TEST_F(TrainerTest, ForwardsLvalueConstructorArgument) {
    double value = 17.5;

    trainer.addMetric<ConfigurableMetric>("custom", value);

    auto results = trainer.evaluate(loader);

    EXPECT_DOUBLE_EQ(results.at("custom"), 17.5);
}

// ============================================================
// Training
// ============================================================

TEST_F(TrainerTest, TrainingUpdatesParameters) {
    auto before = snapshotParameters();

    trainer.fit(loader, 1);

    auto after = snapshotParameters();

    EXPECT_NE(before, after);
}

TEST_F(TrainerTest, TrainingReducesLoss) {
    double initial_loss = trainer.evaluate(loader).at("loss");

    trainer.fit(loader, 10);

    double final_loss = trainer.evaluate(loader).at("loss");

    EXPECT_LT(final_loss, initial_loss);
}

TEST_F(TrainerTest, TrainingProducesFiniteLoss) {
    auto history = trainer.fit(loader, 5);

    for (const auto& epoch : history) {
        EXPECT_TRUE(std::isfinite(epoch.at("loss")));
        EXPECT_GE(epoch.at("loss"), 0.0);
    }
}

TEST_F(TrainerTest, ZeroEpochsDoesNotModifyParameters) {
    auto before = snapshotParameters();

    auto history = trainer.fit(loader, 0);

    EXPECT_TRUE(history.empty());
    EXPECT_EQ(before, snapshotParameters());
}

// ============================================================
// Training history
// ============================================================

TEST_F(TrainerTest, HistoryContainsOneEntryPerEpoch) {
    auto history = trainer.fit(loader, 5);

    EXPECT_EQ(history.size(), 5u);
}

TEST_F(TrainerTest, HistoryContainsLoss) {
    auto history = trainer.fit(loader, 3);

    for (const auto& epoch : history) {
        EXPECT_TRUE(epoch.contains("loss"));
    }
}

TEST_F(TrainerTest, HistoryContainsRegisteredMetrics) {
    trainer.addMetric<SampleCountMetric>("samples");

    auto history = trainer.fit(loader, 3);

    for (const auto& epoch : history) {
        EXPECT_TRUE(epoch.contains("loss"));
        EXPECT_TRUE(epoch.contains("samples"));
        EXPECT_DOUBLE_EQ(epoch.at("samples"), 5.0);
    }
}

TEST_F(TrainerTest, HistoryIsIndependentBetweenFitCalls) {
    auto first_history = trainer.fit(loader, 3);
    auto second_history = trainer.fit(loader, 2);

    EXPECT_EQ(first_history.size(), 3u);
    EXPECT_EQ(second_history.size(), 2u);
}

TEST_F(TrainerTest, HistoryWorksWithoutRegisteredMetrics) {
    auto history = trainer.fit(loader, 2);

    for (const auto& epoch : history) {
        EXPECT_EQ(epoch.size(), 1u);
        EXPECT_TRUE(epoch.contains("loss"));
    }
}

// ============================================================
// Evaluation
// ============================================================

TEST_F(TrainerTest, EvaluationReturnsLoss) {
    auto results = trainer.evaluate(loader);

    EXPECT_TRUE(results.contains("loss"));
    EXPECT_TRUE(std::isfinite(results.at("loss")));
    EXPECT_GE(results.at("loss"), 0.0);
}

TEST_F(TrainerTest, EvaluationDoesNotModifyParameters) {
    auto before = snapshotParameters();

    trainer.evaluate(loader);

    EXPECT_EQ(before, snapshotParameters());
}

TEST_F(TrainerTest, EvaluationDoesNotModifyGradients) {
    trainer.fit(loader, 1);

    auto before = snapshotGradients();

    trainer.evaluate(loader);

    EXPECT_EQ(before, snapshotGradients());
}

TEST_F(TrainerTest, RepeatedEvaluationProducesSameLoss) {
    auto first = trainer.evaluate(loader);
    auto second = trainer.evaluate(loader);

    EXPECT_DOUBLE_EQ(first.at("loss"), second.at("loss"));
}

TEST_F(TrainerTest, EvaluationReturnsSampleWeightedLoss) {
    // The dataset contains five samples with batch size two.
    // The last batch contains only one sample.
    double total_loss = 0.0;
    std::size_t total_samples = 0;

    {
        NoGradGuard guard;

        for (auto [x, y] : loader) {
            Tensor output = model(x);
            Tensor loss = criterion(output, y);

            std::size_t batch_size = x.shape()[0];

            total_loss += loss.data().as(f64).scalar<double>() * batch_size;

            total_samples += batch_size;
        }
    }

    double expected_loss = total_loss / total_samples;

    auto results = trainer.evaluate(loader);

    EXPECT_NEAR(results.at("loss"), expected_loss, 1e-6);
}

// ============================================================
// Metric lifecycle
// ============================================================

TEST_F(TrainerTest, MetricsResetBetweenTrainingEpochs) {
    trainer.addMetric<SampleCountMetric>("samples");

    auto history = trainer.fit(loader, 3);

    for (const auto& epoch : history) {
        EXPECT_DOUBLE_EQ(epoch.at("samples"), 5.0);
    }
}

TEST_F(TrainerTest, MetricsResetBetweenEvaluations) {
    trainer.addMetric<SampleCountMetric>("samples");

    auto first = trainer.evaluate(loader);
    auto second = trainer.evaluate(loader);

    EXPECT_DOUBLE_EQ(first.at("samples"), 5.0);
    EXPECT_DOUBLE_EQ(second.at("samples"), 5.0);
}

TEST_F(TrainerTest, MetricsResetBetweenTrainingAndEvaluation) {
    trainer.addMetric<SampleCountMetric>("samples");

    trainer.fit(loader, 3);

    auto results = trainer.evaluate(loader);

    EXPECT_DOUBLE_EQ(results.at("samples"), 5.0);
}

TEST_F(TrainerTest, MetricsResetBetweenEvaluationAndTraining) {
    trainer.addMetric<SampleCountMetric>("samples");

    trainer.evaluate(loader);

    auto history = trainer.fit(loader, 1);

    EXPECT_DOUBLE_EQ(history.front().at("samples"), 5.0);
}

// ============================================================
// Gradient mode
// ============================================================

TEST_F(TrainerTest, TrainingMetricUpdatesDisableGradMode) {
    trainer.addMetric<GradModeMetric>("grad_mode");

    auto history = trainer.fit(loader, 2);

    for (const auto& epoch : history) {
        EXPECT_DOUBLE_EQ(epoch.at("grad_mode"), 0.0);
    }
}

TEST_F(TrainerTest, EvaluationMetricUpdatesDisableGradMode) {
    trainer.addMetric<GradModeMetric>("grad_mode");

    auto results = trainer.evaluate(loader);

    EXPECT_DOUBLE_EQ(results.at("grad_mode"), 0.0);
}

TEST_F(TrainerTest, EvaluationDisablesGradModeDuringLoss) {
    bool grad_enabled = false;

    LossFunction checking_loss = [&](const Tensor& predictions, const Tensor& targets) {
        grad_enabled |= GradMode::isEnabled();

        return criterion(predictions, targets);
    };

    Trainer checking_trainer(model, optim, checking_loss, {.verbose = false});

    checking_trainer.evaluate(loader);

    EXPECT_FALSE(grad_enabled);
}

TEST_F(TrainerTest, EvaluationRestoresGradMode) {
    bool previous = GradMode::isEnabled();

    trainer.evaluate(loader);

    EXPECT_EQ(GradMode::isEnabled(), previous);
}

TEST_F(TrainerTest, EvaluationPreservesDisabledGradMode) {
    NoGradGuard guard;

    trainer.evaluate(loader);

    EXPECT_FALSE(GradMode::isEnabled());
}

TEST_F(TrainerTest, TrainingRestoresGradModeAfterMetricUpdates) {
    trainer.addMetric<GradModeMetric>("grad_mode");

    bool previous = GradMode::isEnabled();

    trainer.fit(loader, 2);

    EXPECT_EQ(GradMode::isEnabled(), previous);
}

TEST_F(TrainerTest, EvaluationRestoresGradModeAfterException) {
    LossFunction throwing_loss = [](const Tensor&, const Tensor&) -> Tensor {
        throw std::runtime_error("Loss failed");
    };

    Trainer throwing_trainer(model, optim, throwing_loss, {.verbose = false});

    bool previous = GradMode::isEnabled();

    EXPECT_THROW(throwing_trainer.evaluate(loader), std::runtime_error);

    EXPECT_EQ(GradMode::isEnabled(), previous);
}

// ============================================================
// Logging
// ============================================================

TEST_F(TrainerTest, SilentModeProducesNoOutput) {
    testing::internal::CaptureStdout();

    auto history = trainer.fit(loader, 2);

    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_TRUE(output.empty());
    EXPECT_EQ(history.size(), 2u);
}

TEST_F(TrainerTest, VerboseModePrintsEpochResults) {
    Trainer verbose_trainer(model, optim, criterion, {.verbose = true});

    testing::internal::CaptureStdout();

    verbose_trainer.fit(loader, 2);

    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_NE(output.find("Epoch [1/2]"), std::string::npos);
    EXPECT_NE(output.find("Epoch [2/2]"), std::string::npos);
    EXPECT_NE(output.find("loss"), std::string::npos);
}

TEST_F(TrainerTest, VerboseModePrintsRegisteredMetrics) {
    Trainer verbose_trainer(model, optim, criterion, {.verbose = true});

    verbose_trainer.addMetric<SampleCountMetric>("samples");

    testing::internal::CaptureStdout();

    verbose_trainer.fit(loader, 1);

    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_NE(output.find("samples"), std::string::npos);
}

TEST_F(TrainerTest, EvaluationDoesNotPrint) {
    Trainer verbose_trainer(model, optim, criterion, {.verbose = true});

    testing::internal::CaptureStdout();

    verbose_trainer.evaluate(loader);

    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_TRUE(output.empty());
}

TEST_F(TrainerTest, VerboseDefaultsToTrue) {
    Trainer default_trainer(model, optim, criterion);

    testing::internal::CaptureStdout();

    default_trainer.fit(loader, 1);

    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_NE(output.find("Epoch [1/1]"), std::string::npos);
}
