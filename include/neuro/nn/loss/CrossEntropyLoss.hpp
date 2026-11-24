#pragma once

#include "neuro/Tensor.hpp"
namespace neuro::nn {

class CrossEntropyLoss {
public:
    Tensor operator()(const Tensor& predicted, const std::vector<int>& target) const;
    Tensor operator()(const Tensor& predicted, const Tensor& target) const;
    Tensor operator()(const Tensor& predicted, std::initializer_list<int> target) const {
        return operator()(predicted, std::vector<int>(target));
    }
};

} // namespace neuro::nn
