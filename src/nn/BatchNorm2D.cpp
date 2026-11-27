#include "neuro/nn/BatchNorm2D.hpp"

#include "neuro/Tensor.hpp"
#include "neuro/nn/BatchNormBase.hpp"

#include <af/defines.h>
#include <af/dim4.hpp>
#include <stdexcept>

namespace neuro::nn {

BatchNorm2D::BatchNorm2D(int num_features, double eps, double momentum)
    : BatchNormBase(num_features, eps, momentum) {}

Tensor BatchNorm2D::forward(const Tensor& input) const {
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
    constexpr std::array<int, 3> dims{0, 2, 3};
    return normalize(input, dims);
}

} // namespace neuro::nn