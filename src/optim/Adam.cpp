#include "neuro/optim/Adam.hpp"

#include "neuro/Tensor.hpp"
#include "neuro/optim/AdamBase.hpp"

namespace neuro {

Adam::Adam(std::vector<Tensor> parameters, double lr, double b1, double b2, double eps,
           double weight_decay)
    : AdamBase(std::move(parameters), lr, b1, b2, eps, weight_decay) {};

void Adam::step() {
    stepImpl(false);
};

} // namespace neuro
