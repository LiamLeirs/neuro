#include "neuro/data/transforms/RandomCrop.hpp"

#include "neuro/data/transforms/Normalize.hpp"

#include <af/array.h>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/random.h>
#include <arrayfire.h>
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

    if (input.shape()[3] != 1) {
        throw std::invalid_argument("Expected a single CHW image");
    }

    const af::dim4 padding(0, padding_, padding_, 0);
    Tensor padded_input = input.pad(padding, padding);

    const auto dims = padded_input.shape();

    const dim_t max_y = dims[1] - height_;
    const dim_t max_x = dims[2] - width_;

    if (max_y < 0 || max_x < 0) {
        throw std::out_of_range("Crop exceeds padded image dimensions");
    }

    if (max_y == 0 && max_x == 0) {
        return padded_input;
    }

    float random_values[2];
    af::randu(2, f32).host(random_values);

    const dim_t y = static_cast<dim_t>(random_values[0] * (max_y + 1));
    const dim_t x = static_cast<dim_t>(random_values[1] * (max_x + 1));

    return padded_input.crop({0, y, x, 0}, {dims[0], height_, width_, 1});
}

bool RandomCrop::supportsBatch() const noexcept {
    return true;
}

Tensor RandomCrop::forwardBatch(const Tensor& input) const {
    if (input.data().elements() == 0) {
        throw std::invalid_argument("Tensor is empty");
    }
    af::dim4 padding = {0, 0, padding_, padding_};
    Tensor padded_batch = input.pad(padding, padding);
    const auto dims = padded_batch.shape();

    const dim_t max_y = dims[2] - height_;
    const dim_t max_x = dims[3] - width_;

    if (max_y < 0 || max_x < 0) {
        throw std::out_of_range("Crop exceeds padded image dimensions");
    }

    if (max_y == 0 && max_x == 0) {
        return padded_batch;
    }

    af::array y = af::floor(af::randu(dims[0], f32) * (max_y + 1)).as(s32);
    af::array x = af::floor(af::randu(dims[0], f32) * (max_x + 1)).as(s32);
    Tensor y_offsets(y);
    Tensor x_offsets(x);

    return padded_batch.cropBatch(y_offsets, x_offsets, height_, width_);
}

} // namespace neuro::transforms
