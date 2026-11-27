#include "neuro/nn/BatchNormBase.hpp"

#include "neuro/Tensor.hpp"

#include <af/data.h>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace neuro::nn {

BatchNormBase::BatchNormBase(int num_features, double eps, double momentum)
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

Tensor& BatchNormBase::gamma() {
    return gamma_;
}

Tensor& BatchNormBase::beta() {
    return beta_;
}

std::vector<Tensor> BatchNormBase::parameters() const {
    return {gamma_, beta_};
}

const af::array& BatchNormBase::runningMean() const noexcept {
    return running_mean_;
}

const af::array& BatchNormBase::runningVar() const noexcept {
    return running_var_;
}

Tensor BatchNormBase::reduceMean(const Tensor& input, std::span<const int> dims) const {
    Tensor output = input;
    for (auto dim : dims) {
        output = output.mean(dim);
    }
    return output;
}

Tensor BatchNormBase::normalize(const Tensor& input, std::span<const int> dims) const {
    Tensor centered, var;
    if (isTraining()) {
        std::size_t n = 1;
        for (auto dim : dims) {
            n *= input.shape()[dim];
        }
        if (n < 2) {
            throw std::invalid_argument(
                "Batch Norm requires at least 2 values per channel during training");
        }
        Tensor mean = reduceMean(input, dims);
        centered = input - mean;
        var = reduceMean(centered * centered, dims);
        af::array unbiased_var = var.data() * (n / (n - 1.0));
        running_mean_ = (1 - momentum_) * running_mean_ + momentum_ * mean.data();
        running_var_ = (1 - momentum_) * running_var_ + momentum_ * unbiased_var;
        running_mean_.eval();
        running_var_.eval();
    } else {
        centered = input - Tensor(running_mean_, false);
        var = Tensor(running_var_, false);
    }
    Tensor normalized = centered / (var + eps_).sqrt();
    return gamma_ * normalized + beta_;
}

} // namespace neuro::nn