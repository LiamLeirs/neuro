#pragma once

#include "neuro/data/transforms/Transform.hpp"

namespace neuro::transforms {

class RandomCrop : public Transform {
public:
    RandomCrop(int height, int width, int padding = 0);

    Tensor forward(const Tensor& input) const override;

    bool supportsBatch() const noexcept override;

    Tensor forwardBatch(const Tensor& input) const override;

private:
    int height_;
    int width_;
    int padding_;
};

} // namespace neuro::transforms
