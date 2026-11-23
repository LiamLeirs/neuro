#pragma once

#include "neuro/Tensor.hpp"
namespace neuro::nn {

class CrossEntropyLoss {
public:
    Tensor operator()(const Tensor& predicted, const std::vector<int>& target) const;
};

} // namespace neuro::nn
