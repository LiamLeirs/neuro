#pragma once
#include "neuro/Tensor.hpp"
#include "neuro/nn/Module.hpp"

#include <span>
#include <vector>

namespace neuro::nn {

class BatchNormBase : public Module {
protected:
    explicit BatchNormBase(int num_features, double eps = 1e-5, double momentum = 0.1);
    Tensor reduceMean(const Tensor& input, std::span<const int> dims) const;
    Tensor normalize(const Tensor& input, std::span<const int> dims) const;

public:
    Tensor& gamma();

    Tensor& beta();

    std::vector<Tensor> parameters() const override;

    const af::array& runningMean() const noexcept;

    const af::array& runningVar() const noexcept;

protected:
    int num_features_;
    double eps_;
    double momentum_;
    Tensor gamma_;
    Tensor beta_;

    mutable af::array running_mean_;
    mutable af::array running_var_;
};

} // namespace neuro::nn