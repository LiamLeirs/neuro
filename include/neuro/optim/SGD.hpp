#pragma once
#include "neuro/optim/Optimizer.hpp"

#include <vector>

namespace neuro {

class SGD : public Optimizer {
public:
    SGD(std::vector<Tensor> parameters, double lr, double weight_decay = 0.0);
    void step() override;

private:
    double weight_decay_;
};

} // namespace neuro
