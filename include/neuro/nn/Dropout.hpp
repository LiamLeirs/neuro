#pragma once

#include "neuro/Tensor.hpp"
#include "neuro/nn/Module.hpp"

#include <cmath>
#include <stdexcept>

namespace neuro::nn {

class Dropout : public Module {
public:
    explicit Dropout(double p = 0.5) : p_(p) {
        if (!std::isfinite(p_) || p_ < 0 || p_ > 1) {
            throw std::invalid_argument("Dropout probability must be between 0 and 1");
        }
    };

    Tensor forward(const Tensor& input) const override {
        return isTraining() ? input.dropout(p_) : input;
    }

private:
    double p_;
};

} // namespace neuro::nn