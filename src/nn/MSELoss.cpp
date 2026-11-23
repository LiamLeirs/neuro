#include "../include/neuro/nn/MSELoss.hpp"

#include <stdexcept>

namespace neuro::nn {
Tensor MSELoss::operator()(const Tensor& predicted, const Tensor& target) const {
    if (predicted.shape() != target.shape())
        throw std::invalid_argument("Incompatible shapes for computing loss");

    Tensor diff = predicted - target;

    return (diff * diff).mean();
}

} // namespace neuro::nn
