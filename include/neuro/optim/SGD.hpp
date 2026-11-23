#pragma once
#include "neuro/optim/Optimizer.hpp"

#include <vector>

namespace neuro {

class SGD : public Optimizer {
public:
    SGD(std::vector<Tensor> parameters, double lr);
    void step() override;
};

} // namespace neuro
