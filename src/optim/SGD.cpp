#include "../include/neuro/optim/SGD.hpp"

namespace neuro {

SGD::SGD(std::vector<Tensor> parameters, double lr) : parameters_(std::move(parameters)), lr_(lr) {
    if (lr <= 0)
        throw std::invalid_argument("Learning rate must be greater than zero");
};

void SGD::step() {
    for (auto& param : parameters_) {
        if (!param.impl_->requires_grad_)
            continue;
        param.impl_->data_ -= lr_ * param.impl_->grad_;
    }
};

void SGD::zeroGrad() {
    for (auto& param : parameters_) {
        param.zeroGrad();
    }
};

} // namespace neuro
