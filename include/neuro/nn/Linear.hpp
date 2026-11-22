#pragma once
#include "neuro/Tensor.hpp"
namespace neuro::nn {

class Linear {
public:
    Linear(int in_features, int out_features);

    Tensor forward(const Tensor& input) const;
    Tensor operator()(const Tensor& input) const;

    // Access to trainable parameters
    Tensor& weight();
    Tensor& bias();

private:
    Tensor weight_;
    Tensor bias_;
    int in_features_;
    int out_features_;
};

} // namespace neuro::nn