#pragma once

#include "neuro/data/transforms/Transform.hpp"

namespace neuro::transforms {

class RandomHorizontalFlip : public Transform {
public:
    RandomHorizontalFlip(const float p = 0.5);

    Tensor forward(const Tensor& input) const override;

    bool supportsBatch() const noexcept override;

    Tensor forwardBatch(const Tensor& input) const override;

private:
    float p_;
};

} // namespace neuro::transforms
