#pragma once
#include "neuro/Tensor.hpp"
#include "neuro/optim/Optimizer.hpp"

#include <vector>

namespace neuro {

class Adam : public Optimizer {
public:
    Adam(std::vector<Tensor> parameters, double lr, double b1 = 0.9, double b2 = 0.999,
         double eps = 1e-8, double weight_decay = 0.0);
    void step() override;

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
