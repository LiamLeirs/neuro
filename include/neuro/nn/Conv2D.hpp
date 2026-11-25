#pragma once
#include "neuro/Tensor.hpp"
#include "neuro/nn/Module.hpp"

#include <optional>
namespace neuro::nn {

class Conv2D : public Module {
public:
    Conv2D(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0,
           int dilation = 1, bool bias = true);

    Tensor forward(const Tensor& input) const override;

    // Access to trainable parameters
    Tensor& weights();
    Tensor& bias();
    std::vector<Tensor> parameters() const override;

private:
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    int padding_;
    int dilation_;

    Tensor weights_;
    std::optional<Tensor> bias_;
};

} // namespace neuro::nn