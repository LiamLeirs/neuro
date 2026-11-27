#include "neuro/data/TransformDataset.hpp"

#include "neuro/Tensor.hpp"
#include "neuro/data/Dataset.hpp"

#include <memory>
#include <utility>

namespace neuro {
TransformDataset::TransformDataset(const Dataset& dataset,
                                   std::unique_ptr<transforms::Transform> input_transform,
                                   std::unique_ptr<transforms::Transform> target_transform)
    : dataset_(dataset), input_transform_(std::move(input_transform)),
      target_transform_(std::move(target_transform)) {}

std::size_t TransformDataset::size() const {
    return dataset_.size();
}

Dataset::Sample TransformDataset::get(std::size_t index) const {
    Sample sample = dataset_.get(index);
    if (input_transform_ != nullptr)
        sample.input = input_transform_->forward(sample.input);
    if (target_transform_ != nullptr)
        sample.target = target_transform_->forward(sample.target);
    return sample;
}

std::pair<Tensor, Tensor> TransformDataset::getBatch(std::span<const std::size_t> indices) const {
    const bool input_supports_batch = !input_transform_ || input_transform_->supportsBatch();

    const bool target_supports_batch = !target_transform_ || target_transform_->supportsBatch();

    if (!input_supports_batch || !target_supports_batch) {
        return Dataset::getBatch(indices);
    }

    auto batch = dataset_.getBatch(indices);

    if (input_transform_) {
        batch.first = input_transform_->forwardBatch(batch.first);
    }

    if (target_transform_) {
        batch.second = target_transform_->forwardBatch(batch.second);
    }

    return batch;
}
} // namespace neuro