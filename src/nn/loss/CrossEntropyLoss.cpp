#include "neuro/nn/loss/CrossEntropyLoss.hpp"

#include <stdexcept>

namespace neuro::nn {
Tensor CrossEntropyLoss::operator()(const Tensor& predicted, const std::vector<int>& target) const {
    if (predicted.shape()[0] != target.size() || predicted.shape()[2] != 1 ||
        predicted.shape()[3] != 1)
        throw std::invalid_argument("Incompatible shapes for computing loss");
    Tensor max = predicted.max(1);
    Tensor loss = (max + (predicted - max).exp().sum(1).log() - predicted.gather(target)).mean();
    return loss;
}

} // namespace neuro::nn
