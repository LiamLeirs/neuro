#include "neuro/nn/Linear.hpp"

#include <af/array.h>
#include <af/data.h>
#include <af/random.h>
#include <cmath>
#include <stdexcept>

namespace {
neuro::Tensor glorot_init(int in_features, int out_features) {
    if (in_features <= 0 || out_features <= 0) {
        throw std::invalid_argument("Features must be nonzero int");
    }
    // Glorot Init
    double limit = std::sqrt(6.0 / ((double)in_features + out_features));
    return neuro::Tensor((af::randu(in_features, out_features) * 2 - 1) * limit, true);
}
} // namespace

namespace neuro::nn {

Linear::Linear(int in_features, int out_features)
    : weight_(glorot_init(in_features, out_features)),
      bias_(Tensor(af::constant(0, 1, out_features), true)), in_features_(in_features),
      out_features_(out_features) {};

Tensor& Linear::weight() {
    return weight_;
};

Tensor& Linear::bias() {
    return bias_;
};

Tensor Linear::forward(const Tensor& input) const {
    if (input.data().dims(2) != 1 || input.data().dims(3) != 1) {
        throw std::invalid_argument("Input isn't two dimensional");
    } else if (input.data().dims(1) != in_features_) {
        throw std::invalid_argument("Input features don't match");
    }
    return input.matmul(weight_) + bias_;
};

std::vector<Tensor> Linear::parameters() const {
    return {weight_, bias_};
}

} // namespace neuro::nn
