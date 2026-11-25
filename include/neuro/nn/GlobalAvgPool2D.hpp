#pragma once

#include "neuro/Tensor.hpp"
#include "neuro/nn/Module.hpp"

namespace neuro::nn {

class GlobalAvgPool2D : public Module {
public:
    Tensor forward(const Tensor& input) const override {
        return input.globalAvgPool2d();
    }
};

} // namespace neuro::nn