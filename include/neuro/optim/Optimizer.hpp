#pragma once
#include "neuro/Tensor.hpp"

#include <cmath>
#include <vector>

namespace neuro {

class Optimizer {
public:
    Optimizer(std::vector<Tensor> parameters, double lr)
        : parameters_(std::move(parameters)), lr_(lr) {
        if (lr <= 0)
            throw std::invalid_argument("Learning rate must be greater than zero");
        if (!std::isfinite(lr)) {
            throw std::invalid_argument("Learning rate must be finite");
        }
    };
    virtual void step() = 0;
    void zeroGrad() {
        for (auto& param : parameters_) {
            param.zeroGrad();
        }
    };
    virtual ~Optimizer() = default;

protected:
    af::array& data(Tensor& param) {
        return param.impl_->data_;
    }

    const af::array& grad(const Tensor& param) {
        return param.impl_->grad_;
    }

    bool requiresGrad(const Tensor& param) {
        return param.requires_grad();
    }
    std::vector<Tensor> parameters_;
    double lr_;
};

} // namespace neuro
