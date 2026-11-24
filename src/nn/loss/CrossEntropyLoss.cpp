#include "neuro/nn/loss/CrossEntropyLoss.hpp"

#include "neuro/Tensor.hpp"

#include <af/array.h>
#include <stdexcept>

namespace neuro::nn {
Tensor CrossEntropyLoss::operator()(const Tensor& predicted, const std::vector<int>& target) const {
    af::array idx(target.size(), target.data());
    return operator()(predicted, Tensor(idx));
}

Tensor CrossEntropyLoss::operator()(const Tensor& predicted, const Tensor& target) const {
    if (predicted.shape()[0] != target.shape()[0] || predicted.shape()[2] != 1 ||
        predicted.shape()[3] != 1)
        throw std::invalid_argument("Incompatible shapes for computing loss");
    Tensor max = predicted.max(1);
    Tensor loss = (max + (predicted - max).exp().sum(1).log() - predicted.gather(target)).mean();
    return loss;
};

} // namespace neuro::nn
