#include "neuro/Trainer.hpp"

#include "neuro/GradMode.hpp"
#include "neuro/metrics/Mean.hpp"

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace neuro {
Trainer::Trainer(nn::Module& model, Optimizer& optim, LossFunction criterion, TrainerConfig config)
    : model_(model), optim_(optim), criterion_(std::move(criterion)), config_(config) {
    if (!criterion_) {
        throw std::invalid_argument("Provided criterion is invalid");
    }
}

TrainingHistory Trainer::fit(DataLoader& loader, std::size_t num_epochs) {
    return fitImpl(loader, nullptr, num_epochs);
}

TrainingHistory Trainer::fit(DataLoader& train_loader, DataLoader& val_loader,
                             std::size_t num_epochs) {
    return fitImpl(train_loader, &val_loader, num_epochs);
}

TrainingHistory Trainer::fitImpl(DataLoader& train_loader, DataLoader* val_loader,
                                 std::size_t num_epochs) {

    validateSchedulerMonitor(val_loader != nullptr);

    TrainingHistory history;
    history.reserve(num_epochs);

    for (std::size_t epoch = 0; epoch < num_epochs; epoch++) {
        const double current_lr = optim_.learningRate();

        MetricResults results = trainEpoch(train_loader);
        results["lr"] = current_lr;

        if (val_loader) {
            MetricResults val_results = evaluate(*val_loader);
            for (const auto& [name, value] : val_results) {
                results["val_" + name] = value;
            }
        }

        if (scheduler_) {
            scheduler_->step();
        } else if (plateau_scheduler_) {
            plateau_scheduler_->step(results.at(scheduler_monitor_));
        }

        if (config_.verbose) {
            std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs << "]";
            for (const auto& [name, value] : results) {
                std::cout << " - " << name << ": " << value << " ";
            }
            std::cout << '\n';
        }

        history.push_back(std::move(results));
    }
    return history;
}

MetricResults Trainer::trainEpoch(DataLoader& loader) {
    model_.train();

    metrics::Mean mean;

    resetMetrics();

    for (auto [x, y] : loader) {
        optim_.zeroGrad();

        Tensor output = model_(x);
        Tensor loss = criterion_(output, y);

        loss.backward();
        optim_.step();

        const std::size_t batch_size = x.shape()[0];

        mean.update(loss, batch_size);
        updateMetrics(output, y);
    }

    auto results = computeMetrics();

    results["loss"] = mean.compute();

    return results;
}

MetricResults Trainer::evaluate(DataLoader& loader) {
    model_.eval();

    metrics::Mean mean;
    NoGradGuard guard;

    resetMetrics();

    for (auto [x, y] : loader) {
        const std::size_t batch_size = x.shape()[0];
        Tensor output = model_(x);
        Tensor loss = criterion_(output, y);
        mean.update(loss, batch_size);
        updateMetrics(output, y);
    }
    auto results = computeMetrics();

    results["loss"] = mean.compute();

    return results;
};

void Trainer::addMetric(std::string metric_name, std::unique_ptr<metrics::Metric> metric) {
    if (metric_name.empty() || !metric) {
        throw std::invalid_argument("Invalid metric supplied");
    }
    if (metrics_.contains(metric_name)) {
        throw std::invalid_argument("Metric already added");
    }
    if (metric_name == "loss") {
        throw std::invalid_argument("Trainer tracks loss internally");
    }
    if (metric_name == "lr") {
        throw std::invalid_argument("Trainer tracks learning rate internally");
    }
    if (metric_name.starts_with("val_")) {
        throw std::invalid_argument("Trainer tracks validation metrics internally");
    }
    metrics_.emplace(std::move(metric_name), std::move(metric));
};

void Trainer::resetMetrics() {
    for (auto& [name, metric] : metrics_) {
        metric->reset();
    }
}

void Trainer::updateMetrics(const Tensor& predictions, const Tensor& targets) {
    NoGradGuard guard;

    for (auto& [name, metric] : metrics_) {
        metric->update(predictions, targets);
    }
}

MetricResults Trainer::computeMetrics() const {
    MetricResults results;
    for (auto& [name, metric] : metrics_) {
        results[name] = metric->compute();
    }
    return results;
}

void Trainer::setScheduler(LRScheduler& scheduler) {
    if (&scheduler.optimizer() != &optim_) {
        throw std::invalid_argument("Scheduler and trainer optimizers differ");
    }
    plateau_scheduler_ = nullptr;
    scheduler_ = &scheduler;
    scheduler_monitor_.clear();
}

void Trainer::setScheduler(ReduceLROnPlateau& scheduler, std::string monitor) {
    if (&scheduler.optimizer() != &optim_) {
        throw std::invalid_argument("Scheduler and trainer optimizers differ");
    }
    if (monitor.empty()) {
        throw std::invalid_argument("Monitor metric is empty");
    }
    plateau_scheduler_ = &scheduler;
    scheduler_monitor_ = std::move(monitor);
    scheduler_ = nullptr;
}

void Trainer::clearScheduler() noexcept {
    scheduler_ = nullptr;
    plateau_scheduler_ = nullptr;
    scheduler_monitor_.clear();
}

void Trainer::validateSchedulerMonitor(bool has_validation) const {
    if (!plateau_scheduler_) {
        return;
    }

    std::string metric = scheduler_monitor_;

    if (metric.starts_with("val_")) {
        if (!has_validation) {
            throw std::invalid_argument("No validation loader present");
        }

        metric = metric.substr(4);
    }

    if (metric != "loss" && !metrics_.contains(metric)) {
        throw std::invalid_argument("Metric not tracked in trainer");
    }
}

} // namespace neuro