#pragma once

#include "neuro/Tensor.hpp"

#include <cstddef>

namespace neuro::metrics {

class Mean {
public:
    void update(double value, std::size_t weight = 1);
    void update(const Tensor& value, std::size_t weight = 1);
    double compute() const;
    void reset();

private:
    double weighted_sum_ = 0.0;
    std::size_t total_weight_ = 0;
};

} // namespace neuro::metrics
