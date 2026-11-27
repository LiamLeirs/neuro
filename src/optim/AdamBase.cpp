#include "neuro/optim/AdamBase.hpp"

#include "neuro/Tensor.hpp"
#include "neuro/optim/Optimizer.hpp"

#include <af/array.h>
#include <af/data.h>
#include <arrayfire.h>
#include <cmath>

namespace neuro {

AdamBase::AdamBase(std::vector<Tensor> parameters, double lr, double b1, double b2, double eps,
                   double weight_decay)
    : Optimizer(std::move(parameters), lr), b1_(b1), b2_(b2), eps_(eps),
      weight_decay_(weight_decay) {
    if (b1 >= 1 || b1 < 0 || b2 >= 1 || b2 < 0) {
        throw std::invalid_argument("Beta values must be in [0, 1)");
    }
    if (eps <= 0) {
        throw std::invalid_argument("Eps must be positive");
    }
    if (!std::isfinite(b1) || !std::isfinite(b2) || !std::isfinite(eps)) {
        throw std::invalid_argument("Eps, b1 and b2 must be finite");
    }
    if (weight_decay_ < 0 || !std::isfinite(weight_decay_)) {
        throw std::invalid_argument("Weight decay must be non-negative and finite");
    }
    for (const Tensor& param : parameters_) {
        m_.push_back(af::constant(0, param.shape(), param.data().type()));
        v_.push_back(af::constant(0, param.shape(), param.data().type()));
    }
};

void AdamBase::stepImpl(bool decoupled_weight_decay) {
    t_ += 1;
    double d1 = 1 - std::pow(b1_, t_);
    double d2 = 1 - std::pow(b2_, t_);
    for (std::size_t i = 0; i < parameters_.size(); i++) {
        if (!requiresGrad(parameters_[i]))
            continue;

        af::array effective_grad = grad(parameters_[i]);

        if (weight_decay_ != 0.0 && !decoupled_weight_decay) {
            effective_grad = effective_grad + weight_decay_ * data(parameters_[i]);
        }

        m_[i] = b1_ * m_[i] + (1 - b1_) * effective_grad;
        v_[i] = b2_ * v_[i] + (1 - b2_) * effective_grad * effective_grad;

        af::array m_t = (m_[i] / d1);
        af::array v_t = (v_[i] / d2);

        if (weight_decay_ != 0.0 && decoupled_weight_decay) {
            data(parameters_[i]) = data(parameters_[i]) * (1 - lr_ * weight_decay_);
        }

        data(parameters_[i]) -= lr_ * m_t / (af::sqrt(v_t) + eps_);
    }
}

} // namespace neuro
