#pragma once

#include "neuro/Tensor.hpp"
namespace neuro {

class Dataset {
public:
    struct Sample {
        Tensor input;
        Tensor target;
    };

    virtual ~Dataset() = default;

    virtual std::size_t size() const = 0;

    virtual Sample get(std::size_t index) const = 0;
};
} // namespace neuro
