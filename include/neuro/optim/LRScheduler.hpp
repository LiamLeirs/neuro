#pragma once

#include "neuro/optim/Optimizer.hpp"

#include <cstddef>
namespace neuro {
class LRScheduler {
public:
    explicit LRScheduler(Optimizer& optimizer)
        : optimizer_(optimizer), base_lr_(optimizer.learningRate()) {}

    virtual void step() = 0;

    virtual ~LRScheduler() = default;

protected:
    Optimizer& optimizer_;
    double base_lr_;
    std::size_t last_epoch_ = 0;
};
} // namespace neuro