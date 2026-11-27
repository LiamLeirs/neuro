#include "neuro/optim/SGD.hpp"

#include "neuro/optim/Optimizer.hpp"

#include <cmath>
#include <stdexcept>

namespace neuro {

SGD::SGD(std::vector<Tensor> parameters, double lr, double weight_decay)
    : Optimizer(std::move(parameters), lr), weight_decay_(weight_decay) {
    if (weight_decay_ < 0 || !std::isfinite(weight_decay_)) {
        throw std::invalid_argument("Weight decay must be non-negative and finite");
    }
};

void SGD::step() {
    for (auto& param : parameters_) {
        if (!requiresGrad(param))
            continue;
        if (weight_decay_ == 0.0) {
            data(param) -= lr_ * grad(param);
        } else {
            data(param) -= lr_ * (grad(param) + weight_decay_ * data(param));
        }
    }
};
} // namespace neuro
