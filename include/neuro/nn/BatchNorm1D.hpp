#pragma once

#include "neuro/nn/BatchNormBase.hpp"

namespace neuro::nn {

class BatchNorm1D : public BatchNormBase {
public:
    explicit BatchNorm1D(int num_features, double eps = 1e-5, double momentum = 0.1);

    Tensor forward(const Tensor& input) const override;
};

} // namespace neuro::nn