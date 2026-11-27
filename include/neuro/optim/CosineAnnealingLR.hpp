#pragma once

#include "neuro/optim/LRScheduler.hpp"
#include "neuro/optim/Optimizer.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>

namespace neuro {
class CosineAnnealingLR : public LRScheduler {
public:
    CosineAnnealingLR(Optimizer& optimizer, std::size_t t_max, double eta_min = 0.0)
        : LRScheduler(optimizer), eta_min_(eta_min), t_max_(t_max) {
        if (!std::isfinite(eta_min_) || eta_min < 0.0 || t_max_ == 0.0) {
            throw std::invalid_argument(
                "T-max should be greater than zero, eta-min should be non-negative and finite");
        }
        if (eta_min_ > base_lr_) {
            throw std::invalid_argument("Eta-min can't be greater than base lr");
        }
    }

    void step() override {
        ++last_epoch_;

        const auto t = std::min(last_epoch_, t_max_);

        const double new_lr = eta_min_ + 0.5 * (base_lr_ - eta_min_) *
                                             (1.0 + std::cos(std::numbers::pi * t / t_max_));

        optimizer_.setLearningRate(new_lr);
    }

private:
    double eta_min_;
    std::size_t t_max_;
};
} // namespace neuro