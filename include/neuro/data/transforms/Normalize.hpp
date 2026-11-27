#pragma once

#include "neuro/data/transforms/Transform.hpp"

#include <vector>

namespace neuro::transforms {

class Normalize : public Transform {
public:
    Normalize(const std::vector<float>& means, const std::vector<float>& stds);

    Tensor forward(const Tensor& input) const override;

    bool supportsBatch() const noexcept override;

    Tensor forwardBatch(const Tensor& input) const override;

private:
    Tensor normalize(const Tensor& input, bool batched) const;
    Tensor means_;
    Tensor stds_;
};

} // namespace neuro::transforms
