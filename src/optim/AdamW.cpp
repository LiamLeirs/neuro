#include "neuro/optim/AdamW.hpp"

#include "neuro/Tensor.hpp"
#include "neuro/optim/AdamBase.hpp"

namespace neuro {

AdamW::AdamW(std::vector<Tensor> parameters, double lr, double b1, double b2, double eps,
             double weight_decay)
    : AdamBase(std::move(parameters), lr, b1, b2, eps, weight_decay) {};

void AdamW::step() {
    stepImpl(true);
};

} // namespace neuro
