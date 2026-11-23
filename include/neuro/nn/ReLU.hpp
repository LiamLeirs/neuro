#pragma once
#include "neuro/Tensor.hpp"
#include "neuro/nn/Module.hpp"
namespace neuro::nn {

class ReLU : public Module {
public:
    Tensor forward(const Tensor& input) const override {
        return input.relu();
    };
};

} // namespace neuro::nn