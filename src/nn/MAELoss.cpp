#include "../include/neuro/nn/MAELoss.hpp"

#include <stdexcept>

namespace neuro::nn {
Tensor MAELoss::operator()(const Tensor& predicted, const Tensor& target) const {
    if (predicted.shape() != target.shape())
        throw std::invalid_argument("Incompatible shapes for computing loss");

    return (predicted - target).abs().mean();
}

} // namespace neuro::nn
