#include "neuro/data/transforms/RandomCrop.hpp"

#include "neuro/data/transforms/Normalize.hpp"

#include <af/array.h>
#include <af/dim4.hpp>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace neuro::transforms {
RandomCrop::RandomCrop(int height, int width, int padding)
    : height_(height), width_(width), padding_(padding) {
    if (height <= 0 || width <= 0 || padding < 0) {
        throw std::invalid_argument(
            "Width and height must be positive, padding must be non-negative");
    }
}

Tensor RandomCrop::forward(const Tensor& input) const {
    if (input.data().elements() == 0) {
        throw std::invalid_argument("Tensor is empty");
    }
    af::dim4 padding = {0, padding_, padding_, 0};
    Tensor padded_input = input.pad(padding, padding);
    af::dim4 original_shape = input.shape();
    af::dim4 padded_shape = padded_input.shape();
    int max_y = padded_shape[1] - height_;
    int max_x = padded_shape[2] - width_;
}

bool RandomCrop::supportsBatch() const noexcept {
    return true;
}

Tensor RandomCrop::forwardBatch(const Tensor& input) const {
    return input;
}

} // namespace neuro::transforms
