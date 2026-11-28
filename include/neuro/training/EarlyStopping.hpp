#pragma once

#include "neuro/training/PlateauMode.hpp"

#include <cmath>
#include <optional>
#include <stdexcept>

namespace neuro {
class EarlyStopping {
public:
    explicit EarlyStopping(PlateauMode mode = PlateauMode::Min, std::size_t patience = 10,
                           double threshold = 1e-4)
        : mode_(mode), patience_(patience), threshold_(threshold) {
        if (mode_ != PlateauMode::Min && mode_ != PlateauMode::Max) {
            throw std::invalid_argument("Invalid plateau mode");
        }

        if (!std::isfinite(threshold_) || threshold_ < 0.0) {
            throw std::invalid_argument("Threshold must be finite and non-negative");
        }
    }

    void step(double metric) {
        if (!std::isfinite(metric)) {
            throw std::invalid_argument("Metric must be finite");
        }

        if (stopped_) {
            return;
        }

        if (isBetter(metric)) {
            best_ = metric;
            bad_epochs_ = 0;
        } else {
            bad_epochs_ += 1;

            if (bad_epochs_ >= patience_) {
                stopped_ = true;
            }
        }
    }

    bool shouldStop() const noexcept {
        return stopped_;
    }

    void reset() noexcept {
        stopped_ = false;
        bad_epochs_ = 0;
        best_.reset();
    }

private:
    PlateauMode mode_;
    std::size_t patience_;
    double threshold_;
    std::optional<double> best_;
    std::size_t bad_epochs_ = 0;
    bool stopped_ = false;

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