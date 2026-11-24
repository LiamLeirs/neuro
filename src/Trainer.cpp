#include "neuro/Trainer.hpp"

#include "neuro/GradMode.hpp"
#include "neuro/metrics/Mean.hpp"

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace neuro {
Trainer::Trainer(nn::Module& model, Optimizer& optim, LossFunction criterion)
    : model_(model), optim_(optim), criterion_(std::move(criterion)) {
    if (!criterion_) {
        throw std::invalid_argument("Provided criterion is invalid");
    }
}

void Trainer::fit(DataLoader& loader, std::size_t num_epochs) {
    for (std::size_t epoch = 0; epoch < num_epochs; epoch++) {
        std::map<std::string, double> results = trainEpoch(loader);
        std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs << "]";
        for (auto& [name, value] : results) {
            std::cout << " - " << name << ": " << value << " ";
        }
        std::cout << '\n';
    }
}

std::map<std::string, double> Trainer::trainEpoch(DataLoader& loader) {
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

std::map<std::string, double> Trainer::evaluate(DataLoader& loader) {
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

std::map<std::string, double> Trainer::computeMetrics() const {
    std::map<std::string, double> results;
    for (auto& [name, metric] : metrics_) {
        results[name] = metric->compute();
    }
    return results;
}

} // namespace neuro