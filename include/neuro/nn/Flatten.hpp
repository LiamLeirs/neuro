#pragma once

#include "neuro/Tensor.hpp"
#include "neuro/nn/Module.hpp"

namespace neuro::nn {

class Flatten : public Module {
public:
    Tensor forward(const Tensor& input) const override {
        return input.flatten();
    }
};

} // namespace neuro::nn