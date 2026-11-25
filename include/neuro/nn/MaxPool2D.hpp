#pragma once

#include "neuro/Tensor.hpp"
#include "neuro/nn/Module.hpp"

#include <stdexcept>

namespace neuro::nn {

class MaxPool2D : public Module {
public:
    explicit MaxPool2D(int window_size, int stride, int padding = 0)
        : window_size_(window_size), stride_(stride), padding_(padding) {
        if (window_size_ < 1)
            throw std::invalid_argument("Window size must be positive");

        if (stride_ < 1)
            throw std::invalid_argument("Stride must be positive");

        if (padding_ < 0)
            throw std::invalid_argument("Padding must be non-negative");

        if (padding_ > window_size_ / 2)
            throw std::invalid_argument("Padding must not exceed half the window size");
    }

    // Default stride equals window size.
    explicit MaxPool2D(int window_size) : MaxPool2D(window_size, window_size, 0) {}

    Tensor forward(const Tensor& input) const override {
        return input.maxpool2d(window_size_, stride_, padding_);
    }

private:
    int window_size_;
    int stride_;
    int padding_;
};

} // namespace neuro::nn