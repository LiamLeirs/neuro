#include "neuro/metrics/Mean.hpp"

#include <stdexcept>

namespace neuro::metrics {

void Mean::update(double value, std::size_t weight) {
    if (weight == 0)
        return;

    weighted_sum_ += value * weight;
    total_weight_ += weight;
}

void Mean::update(const Tensor& value, std::size_t weight) {
    if (!value.data().isscalar()) {
        throw std::invalid_argument("Tensor is not a scalar");
    }
    if (weight == 0)
        return;
    update(value.data().as(f64).scalar<double>(), weight);
}

double Mean::compute() const {
    if (total_weight_ == 0)
        return 0.0;

    return weighted_sum_ / total_weight_;
}

void Mean::reset() {
    weighted_sum_ = 0;
    total_weight_ = 0;
}

} // namespace neuro::metrics