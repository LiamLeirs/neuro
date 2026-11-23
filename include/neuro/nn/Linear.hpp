#pragma once
#include "neuro/Tensor.hpp"
#include "neuro/nn/Module.hpp"
namespace neuro::nn {

class Linear : public Module {
public:
    Linear(int in_features, int out_features);

    Tensor forward(const Tensor& input) const override;

    // Access to trainable parameters
    Tensor& weight();
    Tensor& bias();
    std::vector<Tensor> parameters() const override;

private:
    Tensor weight_;
    Tensor bias_;
    int in_features_;
    int out_features_;
};

} // namespace neuro::nn