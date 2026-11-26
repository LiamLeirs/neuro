#include "neuro/nn/BatchNorm1D.hpp"

#include "neuro/Tensor.hpp"

#include <af/data.h>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace neuro::nn {

BatchNorm1D::BatchNorm1D(int num_features, double eps, double momentum)
    : num_features_(num_features), eps_(eps), momentum_(momentum) {
    if (num_features_ <= 0) {
        throw std::invalid_argument("Num features must be greater than zero");
    }
    if (eps_ <= 0 || !std::isfinite(eps_)) {
        throw std::invalid_argument("Eps must be greater than zero and finite");
    }
    if (momentum_ < 0 || momentum_ > 1 || !std::isfinite(momentum_)) {
        throw std::invalid_argument("Momentum must be in [0, 1]");
    }
    gamma_ = Tensor(af::constant(1, af::dim4(1, num_features_), f32), true);
    beta_ = Tensor(af::constant(0, af::dim4(1, num_features_), f32), true);
    running_mean_ = af::constant(0, af::dim4(1, num_features_), f32);
    running_var_ = af::constant(1, af::dim4(1, num_features_), f32);
}

Tensor BatchNorm1D::forward(const Tensor& input) const {
    if (input.data().elements() == 0) {
        throw std::invalid_argument("Tensor is empty");
    }
    af::dtype type = input.data().type();
    af::dim4 shape = input.shape();
    if (type != f32) {
        throw std::invalid_argument("Batch Norm requires f32 input");
    }
    if (num_features_ != shape[1]) {
        throw std::invalid_argument("Mismatch between feature count");
    }
    if (shape[2] != 1 || shape[3] != 1) {
        throw std::invalid_argument("Input must be 2-dimensional");
    }
    if (isTraining() && shape[0] < 2) {
        throw std::invalid_argument("Batch Norm requires at least 2 samples during training");
    }
    Tensor mean, var;
    if (isTraining()) {
        mean = input.mean(0);
        var = input.var(0);
        const double n = static_cast<double>(shape[0]);
        af::array unbiased_var = var.data() * (n / (n - 1.0));
        running_mean_ = (1 - momentum_) * running_mean_ + momentum_ * mean.data();
        running_var_ = (1 - momentum_) * running_var_ + momentum_ * unbiased_var;
        running_mean_.eval();
        running_var_.eval();
    } else {
        mean = Tensor(running_mean_, false);
        var = Tensor(running_var_, false);
    }
    Tensor normalized = (input - mean) / (var + eps_).sqrt();
    return gamma_ * normalized + beta_;
}

Tensor& BatchNorm1D::gamma() {
    return gamma_;
}

Tensor& BatchNorm1D::beta() {
    return beta_;
}

std::vector<Tensor> BatchNorm1D::parameters() const {
    return {gamma_, beta_};
}

const af::array& BatchNorm1D::runningMean() const noexcept {
    return running_mean_;
}

const af::array& BatchNorm1D::runningVar() const noexcept {
    return running_var_;
}

} // namespace neuro::nn