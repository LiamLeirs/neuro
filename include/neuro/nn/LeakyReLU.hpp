#pragma once
#include "neuro/Tensor.hpp"
#include "neuro/nn/Module.hpp"

#include <cmath>
#include <stdexcept>
namespace neuro::nn {

class LeakyReLU : public Module {
public:
    LeakyReLU(float negativeSlope = 0.01f) : negativeSlope_(negativeSlope) {
        if (!std::isfinite(negativeSlope))
            throw std::invalid_argument("Negative slope must be finite");
    };
    Tensor forward(const Tensor& input) const override {
        return input.leakyRelu(negativeSlope_);
    };

private:
    float negativeSlope_;
};

} // namespace neuro::nn