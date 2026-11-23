#include "../include/neuro/optim/SGD.hpp"

#include "neuro/optim/Optimizer.hpp"

namespace neuro {

SGD::SGD(std::vector<Tensor> parameters, double lr) : Optimizer(std::move(parameters), lr) {};

void SGD::step() {
    for (auto& param : parameters_) {
        if (!requiresGrad(param))
            continue;
        data(param) -= lr_ * grad(param);
    }
};
} // namespace neuro
