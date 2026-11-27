#pragma once

#include "neuro/Tensor.hpp"

namespace neuro::transforms {

class Transform {
public:
    virtual Tensor forward(const Tensor& input) const = 0;

    Tensor operator()(const Tensor& input) const {
        return forward(input);
    };

    virtual ~Transform() = default;
};

} // namespace neuro::transforms