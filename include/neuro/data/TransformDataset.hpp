#pragma once

#include "neuro/data/Dataset.hpp"
#include "neuro/data/transforms/Transform.hpp"

#include <cstddef>

namespace neuro {

class TransformDataset : public Dataset {
public:
    TransformDataset(const Dataset& dataset,
                     std::unique_ptr<transforms::Transform> input_transform = {},
                     std::unique_ptr<transforms::Transform> target_transform = {});

    std::size_t size() const override;

    Sample get(std::size_t index) const override;

    std::pair<Tensor, Tensor> getBatch(std::span<const std::size_t> indices) const override;

private:
    const Dataset& dataset_;
    std::unique_ptr<transforms::Transform> input_transform_;
    std::unique_ptr<transforms::Transform> target_transform_;
};
} // namespace neuro
