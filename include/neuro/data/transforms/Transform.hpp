#pragma once

#include "neuro/Tensor.hpp"

#include <stdexcept>

namespace neuro::transforms {

class Transform {
public:
    virtual Tensor forward(const Tensor& input) const = 0;

    Tensor operator()(const Tensor& input) const {
        return forward(input);
    };

    virtual bool supportsBatch() const noexcept {
        return false;
    };

    virtual Tensor forwardBatch(const Tensor& input) const {
        throw std::runtime_error("Transform doesn't support batch forwarding");
    };

    virtual ~Transform() = default;
};

} // namespace neuro::transforms