#include "neuro/data/transforms/Normalize.hpp"

#include <af/array.h>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace neuro::transforms {
Normalize::Normalize(const std::vector<float>& means, const std::vector<float>& stds) {
    if (means.empty() || stds.empty()) {
        throw std::invalid_argument("Means and stds must be non-empty");
    }
    if (means.size() != stds.size()) {
        throw std::invalid_argument("Means and stds must be same size");
    }
    for (std::size_t i = 0; i < means.size(); i++) {
        if (!std::isfinite(means[i]) || !std::isfinite(stds[i])) {
            throw std::invalid_argument("Means and stds must be finite");
        }
        if (stds[i] <= 0) {
            throw std::invalid_argument("Stds must be greater than zero");
        }
    }
    means_ = Tensor(af::array(means.size(), means.data()));
    stds_ = Tensor(af::array(stds.size(), stds.data()));
}

Tensor Normalize::forward(const Tensor& input) const {
    return normalize(input, false);
}

bool Normalize::supportsBatch() const noexcept {
    return true;
}

Tensor Normalize::forwardBatch(const Tensor& input) const {
    return normalize(input, true);
}

Tensor Normalize::normalize(const Tensor& input, bool batched) const {
    const auto dtype = input.data().type();

    if (dtype != f32 && dtype != f64) {
        throw std::invalid_argument("Normalize requires floating-point input");
    }

    const int dim = batched ? 1 : 0;
    if (input.data().elements() == 0) {
        throw std::invalid_argument("Input is empty");
    }

    if (input.shape()[dim] != means_.shape()[0]) {
        throw std::invalid_argument("Input channel count does not match normalization statistics");
    }

    Tensor means = means_;
    Tensor stds = stds_;
    if (batched) {
        means = means.reshape({1, input.shape()[1]});
        stds = stds.reshape({1, input.shape()[1]});
    }
    return (input - means) / stds;
}

} // namespace neuro::transforms
