#pragma once

#include "neuro/Tensor.hpp"
#include "neuro/nn/Module.hpp"

#include <stdexcept>

namespace neuro::nn {

class AvgPool2D : public Module {
public:
    explicit AvgPool2D(int window_size, int stride, int padding = 0, bool count_include_pad = true)
        : window_size_(window_size), stride_(stride), padding_(padding),
          count_include_pad_(count_include_pad) {
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
    explicit AvgPool2D(int window_size) : AvgPool2D(window_size, window_size, 0, true) {}

    Tensor forward(const Tensor& input) const override {
        return input.avgpool2d(window_size_, stride_, padding_, count_include_pad_);
    }

private:
    int window_size_;
    int stride_;
    int padding_;
    bool count_include_pad_;
};

} // namespace neuro::nn