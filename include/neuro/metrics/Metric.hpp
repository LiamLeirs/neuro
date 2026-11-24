#pragma once

#include "neuro/Tensor.hpp"

namespace neuro::metrics {

class Metric {
public:
    virtual void update(const Tensor& predictions, const Tensor& targets) = 0;
    virtual double compute() const = 0;
    virtual void reset() = 0;
    virtual ~Metric() = default;
};

} // namespace neuro::metrics
