#include "../include/neuro/optim/Adam.hpp"

#include "neuro/Tensor.hpp"
#include "neuro/optim/Optimizer.hpp"

#include <af/array.h>
#include <af/data.h>
#include <arrayfire.h>
#include <cmath>

namespace neuro {

Adam::Adam(std::vector<Tensor> parameters, double lr, double b1, double b2, double eps)
    : Optimizer(std::move(parameters), lr), b1_(b1), b2_(b2), eps_(eps) {
    if (b1 >= 1 || b1 < 0 || b2 >= 1 || b2 < 0) {
        throw std::invalid_argument("Beta values must be in [0, 1)");
    }
    if (eps <= 0) {
        throw std::invalid_argument("Eps must be positive");
    }
    if (!std::isfinite(b1) || !std::isfinite(b2) || !std::isfinite(eps)) {
        throw std::invalid_argument("Eps, b1 and b2 must be finite");
    }
    for (const Tensor& param : parameters_) {
        m_.push_back(af::constant(0, param.shape(), param.data().type()));
        v_.push_back(af::constant(0, param.shape(), param.data().type()));
    }
};

void Adam::step() {
    t_ += 1;
    double d1 = 1 - std::pow(b1_, t_);
    double d2 = 1 - std::pow(b2_, t_);
    for (std::size_t i = 0; i < parameters_.size(); i++) {
        if (!requiresGrad(parameters_[i]))
            continue;

        m_[i] = b1_ * m_[i] + (1 - b1_) * grad(parameters_[i]);
        v_[i] = b2_ * v_[i] + (1 - b2_) * grad(parameters_[i]) * grad(parameters_[i]);

        af::array m_t = (m_[i] / d1);
        af::array v_t = (v_[i] / d2);

        data(parameters_[i]) -= lr_ * m_t / (af::sqrt(v_t) + eps_);
    }
};

} // namespace neuro
