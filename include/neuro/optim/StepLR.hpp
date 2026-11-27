#pragma once

#include "neuro/optim/LRScheduler.hpp"
#include "neuro/optim/Optimizer.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
namespace neuro {
class StepLR : public LRScheduler {
public:
    StepLR(Optimizer& optimizer, std::size_t step_size, double gamma = 0.1)
        : LRScheduler(optimizer), step_size_(step_size), gamma_(gamma) {
        if (step_size_ == 0 || !std::isfinite(gamma) || gamma <= 0.0) {
            throw std::invalid_argument(
                "Step size must be greater than zero, gamma should be positive and finite");
        }
    }

    void step() override {
        last_epoch_ += 1;
        if (last_epoch_ % step_size_ == 0) {
            double current_lr = optimizer_.learningRate();
            optimizer_.setLearningRate(current_lr * gamma_);
        }
    }

private:
    std::size_t step_size_;
    double gamma_;
};
} // namespace neuro