#pragma once

#include "neuro/Tensor.hpp"
#include "neuro/data/DataLoader.hpp"
#include "neuro/metrics/Metric.hpp"
#include "neuro/nn/Module.hpp"
#include "neuro/optim/Optimizer.hpp"

#include <cstddef>
#include <functional>
#include <map>
#include <memory>

namespace neuro {

using LossFunction = std::function<Tensor(const Tensor&, const Tensor&)>;

class Trainer {
public:
    Trainer(nn::Module& model, Optimizer& optim, LossFunction criterion);

    void fit(DataLoader& loader, std::size_t num_epochs);

    std::map<std::string, double> evaluate(DataLoader& loader);

    void addMetric(std::string metric_name, std::unique_ptr<metrics::Metric> metric);

private:
    nn::Module& model_;
    Optimizer& optim_;
    LossFunction criterion_;
    std::map<std::string, std::unique_ptr<metrics::Metric>> metrics_;

    void resetMetrics();

    std::map<std::string, double> trainEpoch(DataLoader& loader);

    void updateMetrics(const Tensor& predictions, const Tensor& targets);

    std::map<std::string, double> computeMetrics() const;
};

} // namespace neuro