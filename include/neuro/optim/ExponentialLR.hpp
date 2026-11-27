#pragma once

#include "neuro/optim/LRScheduler.hpp"
#include "neuro/optim/Optimizer.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
namespace neuro {
class ExponentialLR : public LRScheduler {
public:
    ExponentialLR(Optimizer& optimizer, double gamma) : LRScheduler(optimizer), gamma_(gamma) {
        if (!std::isfinite(gamma) || gamma <= 0.0) {
            throw std::invalid_argument("Gamma should be positive and finite");
        }
    }

    void step() override {
        last_epoch_ += 1;
        const double current_lr = optimizer_.learningRate();
        optimizer_.setLearningRate(current_lr * gamma_);
    }

private:
    double gamma_;
};
} // namespace neuro