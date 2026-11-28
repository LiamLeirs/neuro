#pragma once

#include "neuro/Tensor.hpp"
#include "neuro/data/DataLoader.hpp"
#include "neuro/metrics/Metric.hpp"
#include "neuro/nn/Module.hpp"
#include "neuro/optim/LRScheduler.hpp"
#include "neuro/optim/Optimizer.hpp"
#include "neuro/optim/ReduceLROnPlateau.hpp"
#include "neuro/training/EarlyStopping.hpp"

#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace neuro {

using MetricResults = std::map<std::string, double>;
using TrainingHistory = std::vector<MetricResults>;
using LossFunction = std::function<Tensor(const Tensor&, const Tensor&)>;

struct TrainerConfig {
    bool verbose = true;
};

class Trainer {
public:
    Trainer(nn::Module& model, Optimizer& optim, LossFunction criterion, TrainerConfig config = {});

    TrainingHistory fit(DataLoader& loader, std::size_t num_epochs);

    TrainingHistory fit(DataLoader& train_loader, DataLoader& val_loader, std::size_t num_epochs);

    MetricResults evaluate(DataLoader& loader);

    void addMetric(std::string metric_name, std::unique_ptr<metrics::Metric> metric);

    template <typename MetricType, typename... Args>
        requires std::derived_from<MetricType, metrics::Metric>
    void addMetric(std::string name, Args&&... args) {
        addMetric(std::move(name), std::make_unique<MetricType>(std::forward<Args>(args)...));
    }

    void setScheduler(LRScheduler& scheduler);

    void setScheduler(ReduceLROnPlateau& scheduler, std::string monitor = "val_loss");

    void clearScheduler() noexcept;

    void setEarlyStopping(EarlyStopping& early_stopping, std::string monitor = "val_loss");

    void clearEarlyStopping() noexcept;

private:
    nn::Module& model_;
    Optimizer& optim_;
    LossFunction criterion_;
    TrainerConfig config_;
    std::map<std::string, std::unique_ptr<metrics::Metric>> metrics_;

    void resetMetrics();

    MetricResults trainEpoch(DataLoader& loader);

    void updateMetrics(const Tensor& predictions, const Tensor& targets);

    MetricResults computeMetrics() const;

    TrainingHistory fitImpl(DataLoader& train_loader, DataLoader* val_loader,
                            std::size_t num_epochs);

    LRScheduler* scheduler_ = nullptr;

    ReduceLROnPlateau* plateau_scheduler_ = nullptr;

    std::string scheduler_monitor_;

    void validateMonitor(const std::string& monitor, bool has_validation) const;

    EarlyStopping* early_stopping_ = nullptr;

    std::string early_stopping_monitor_;
};

} // namespace neuro