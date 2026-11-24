#pragma once

#include "neuro/Tensor.hpp"

#include <af/algorithm.h>
#include <stdexcept>

namespace neuro::metrics {

class Accuracy {
public:
    void update(const Tensor& predictions, const Tensor& targets);
    double compute() const;
    void reset();

private:
    std::size_t num_correct_ = 0;
    std::size_t total_samples_ = 0;
};

} // namespace neuro::metrics
