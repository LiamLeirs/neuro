#include "neuro/data/transforms/RandomHorizontalFlip.hpp"

#include "neuro/Tensor.hpp"

#include <af/array.h>
#include <af/data.h>
#include <af/random.h>
#include <cmath>
#include <stdexcept>

namespace neuro::transforms {
RandomHorizontalFlip::RandomHorizontalFlip(const float p) : p_(p) {
    if (p_ < 0 || p_ > 1 || !std::isfinite(p_)) {
        throw std::invalid_argument("Flip probability must in [0, 1]");
    }
}

Tensor RandomHorizontalFlip::forward(const Tensor& input) const {
    if (input.data().elements() == 0) {
        throw std::invalid_argument("Tensor is empty");
    }

    if (p_ == 0.0f) {
        return input;
    }

    if (p_ == 1.0f) {
        return input.flip(2);
    }

    float rand = af::randu(1).scalar<float>();

    if (rand < p_) {
        // Flip width dimension
        return input.flip(2);
    }
    return input;
}

bool RandomHorizontalFlip::supportsBatch() const noexcept {
    return true;
}

Tensor RandomHorizontalFlip::forwardBatch(const Tensor& input) const {
    if (input.data().elements() == 0) {
        throw std::invalid_argument("Tensor is empty");
    }

    if (p_ == 0.0f) {
        return input;
    }

    if (p_ == 1.0f) {
        return input.flip(3);
    }

    Tensor mask = Tensor(af::randu(input.shape()[0]) < p_);
    return Tensor::where(mask, input.flip(3), input);
}

} // namespace neuro::transforms
