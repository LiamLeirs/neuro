#pragma once
#include "neuro/Tensor.hpp"
#include "neuro/nn/Module.hpp"
namespace neuro::nn {

class Sigmoid : public Module {
public:
    Tensor forward(const Tensor& input) const override {
        return input.sigmoid();
    }
};

} // namespace neuro::nn