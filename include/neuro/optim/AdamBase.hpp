#pragma once
#include "neuro/Tensor.hpp"
#include "neuro/optim/Optimizer.hpp"

#include <vector>

namespace neuro {

class AdamBase : public Optimizer {
protected:
    AdamBase(std::vector<Tensor> parameters, double lr, double b1, double b2, double eps,
             double weight_decay);

    void stepImpl(bool decoupled_weight_decay);

private:
    double b1_;
    double b2_;
    double eps_;
    double weight_decay_;

    std::vector<af::array> m_;
    std::vector<af::array> v_;
    std::size_t t_ = 0;
};

} // namespace neuro
