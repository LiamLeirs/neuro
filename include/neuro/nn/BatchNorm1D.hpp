#pragma once
#include "neuro/Tensor.hpp"
#include "neuro/nn/Module.hpp"

#include <vector>

namespace neuro::nn {

class BatchNorm1D : public Module {
public:
    explicit BatchNorm1D(int num_features, double eps = 1e-5, double momentum = 0.1);

    Tensor forward(const Tensor& input) const override;

    Tensor& gamma();

    Tensor& beta();

    std::vector<Tensor> parameters() const override;

    const af::array& runningMean() const noexcept;

    const af::array& runningVar() const noexcept;

private:
    int num_features_;
    double eps_;
    Tensor gamma_;
    Tensor beta_;

    double momentum_;

    mutable af::array running_mean_;
    mutable af::array running_var_;
};

} // namespace neuro::nn