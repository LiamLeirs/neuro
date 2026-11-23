#pragma once
#include "neuro/Tensor.hpp"
#include "neuro/nn/Module.hpp"
namespace neuro::nn {

class Tanh : public Module {
public:
    Tensor forward(const Tensor& input) const override {
        return input.tanh();
    }
};

} // namespace neuro::nn