#pragma once

#include "neuro/Tensor.hpp"
namespace neuro::nn {

class MAELoss {
public:
    Tensor operator()(const Tensor& predicted, const Tensor& target) const;
};

} // namespace neuro::nn
