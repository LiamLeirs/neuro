#pragma once
#include "neuro/Tensor.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace neuro {

class Optimizer {
public:
    Optimizer(std::vector<Tensor> parameters, double lr) : parameters_(std::move(parameters)) {
        setLearningRate(lr);
    };
    virtual void step() = 0;
    void zeroGrad() {
        for (auto& param : parameters_) {
            param.zeroGrad();
        }
    };
    virtual ~Optimizer() = default;

    double learningRate() const noexcept {
        return lr_;
    }

    void setLearningRate(double lr) {
        if (!std::isfinite(lr) || lr < 0.0) {
            throw std::invalid_argument("Learning rate must be finite and non-negative");
        }
        lr_ = lr;
    }

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
