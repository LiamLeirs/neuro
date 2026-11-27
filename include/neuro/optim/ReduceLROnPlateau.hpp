#pragma once

#include "neuro/optim/Optimizer.hpp"

#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>

namespace neuro {

enum class PlateauMode { Min, Max };

class ReduceLROnPlateau {
public:
    ReduceLROnPlateau(Optimizer& optimizer, PlateauMode mode = PlateauMode::Min,
                      double factor = 0.1, std::size_t patience = 10, double threshold = 1e-4,
                      double min_lr = 0.0)
        : optimizer_(optimizer), mode_(mode), factor_(factor), patience_(patience),
          threshold_(threshold), min_lr_(min_lr) {

        if (mode_ != PlateauMode::Min && mode_ != PlateauMode::Max) {
            throw std::invalid_argument("Invalid plateau mode");
        }

        if (!std::isfinite(factor_) || factor_ <= 0.0 || factor_ >= 1.0) {
            throw std::invalid_argument("Factor must be finite and in (0, 1)");
        }

        if (!std::isfinite(threshold_) || threshold_ < 0.0) {
            throw std::invalid_argument("Threshold must be finite and non-negative");
        }

        if (!std::isfinite(min_lr_) || min_lr_ < 0.0) {
            throw std::invalid_argument("Minimum learning rate must be finite and non-negative");
        }

        if (min_lr_ > optimizer_.learningRate()) {
            throw std::invalid_argument(
                "Minimum learning rate cannot exceed initial learning rate");
        }
    }

    void step(double metric) {
        if (!std::isfinite(metric)) {
            throw std::invalid_argument("Metric must be finite");
        }
        last_epoch_ += 1;
        if (isBetter(metric)) {
            best_ = metric;
            bad_epochs_ = 0;
        } else {
            bad_epochs_ += 1;
        }
        if (bad_epochs_ > patience_) {
            const double current_lr = optimizer_.learningRate();
            const double new_lr = std::max(current_lr * factor_, min_lr_);

            if (new_lr < current_lr) {
                optimizer_.setLearningRate(new_lr);
            }

            bad_epochs_ = 0;
        }
    }

private:
    Optimizer& optimizer_;
    PlateauMode mode_;
    double factor_;
    std::size_t patience_;
    double threshold_;
    double min_lr_;

    std::optional<double> best_;
    std::size_t bad_epochs_ = 0;
    std::size_t last_epoch_ = 0;

    bool isBetter(double metric) const {
        if (!best_) {
            return true;
        }

        if (mode_ == PlateauMode::Min) {
            return metric < *best_ - threshold_;
        }
        return metric > *best_ + threshold_;
    }
};

} // namespace neuro