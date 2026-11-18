#pragma once
#include "neuro/Tensor.hpp"

#include <vector>

namespace neuro {

class SGD {
public:
    SGD(std::vector<Tensor> parameters, double lr);
    void step();
    void zeroGrad();

private:
    std::vector<Tensor> parameters_;
    double lr_;
};

} // namespace neuro
