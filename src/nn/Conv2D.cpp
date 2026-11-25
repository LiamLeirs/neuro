#include "neuro/nn/Conv2D.hpp"

#include <af/array.h>
#include <af/data.h>
#include <af/random.h>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {
neuro::Tensor glorot_init(int in_channels, int out_channels, int kernel_size) {
    if (in_channels <= 0 || out_channels <= 0) {
        throw std::invalid_argument("Channel counts must be positive");
    }
    if (kernel_size < 1)
        throw std::invalid_argument("Kernel size must be greater than zero");
    // Glorot Init
    const double fan_in = static_cast<double>(in_channels) * kernel_size * kernel_size;
    const double fan_out = static_cast<double>(out_channels) * kernel_size * kernel_size;
    const double limit = std::sqrt(6.0 / (fan_in + fan_out));
    return neuro::Tensor(
        (af::randu(out_channels, in_channels, kernel_size, kernel_size) * 2 - 1) * limit, true);
}
} // namespace

namespace neuro::nn {

Conv2D::Conv2D(int in_channels, int out_channels, int kernel_size, int stride, int padding,
               int dilation, bool bias)
    : in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size),
      stride_(stride), padding_(padding), dilation_(dilation),
      weights_(glorot_init(in_channels, out_channels, kernel_size)) {
    if (stride_ < 1)
        throw std::invalid_argument("Stride must be greater than zero");
    if (padding_ < 0) {
        throw std::invalid_argument("Padding must be non-negative");
    }
    if (dilation_ < 1) {
        throw std::invalid_argument("Dilation must be greater than zero");
    }
    if (bias) {
        bias_ = Tensor(af::constant(0, 1, out_channels), true);
    }
};

Tensor Conv2D::forward(const Tensor& input) const {
    Tensor result = input.conv2d(weights_, stride_, padding_, dilation_);
    if (bias_) {
        result = result + *bias_;
    }
    return result;
};

Tensor& Conv2D::weights() {
    return weights_;
};
Tensor& Conv2D::bias() {
    if (bias_)
        return *bias_;
    else
        throw std::logic_error("Conv2D has no bias");
};

std::vector<Tensor> Conv2D::parameters() const {
    std::vector<Tensor> params = {weights_};
    if (bias_)
        params.push_back(*bias_);
    return params;
};

} // namespace neuro::nn
