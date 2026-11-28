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

    // Input transform only.
    template <typename T>
        requires std::derived_from<std::remove_cvref_t<T>, transforms::Transform>
    TransformDataset(const Dataset& dataset, T&& input_transform)
        : TransformDataset(
              dataset, std::make_unique<std::remove_cvref_t<T>>(std::forward<T>(input_transform))) {
    }

    // Both input and target transforms.
    template <typename T, typename U>
        requires std::derived_from<std::remove_cvref_t<T>, transforms::Transform> &&
                 std::derived_from<std::remove_cvref_t<U>, transforms::Transform>
    TransformDataset(const Dataset& dataset, T&& input_transform, U&& target_transform)
        : TransformDataset(
              dataset, std::make_unique<std::remove_cvref_t<T>>(std::forward<T>(input_transform)),
              std::make_unique<std::remove_cvref_t<U>>(std::forward<U>(target_transform))) {}

    // Target transform only.
    template <typename T>
        requires std::derived_from<std::remove_cvref_t<T>, transforms::Transform>
    TransformDataset(const Dataset& dataset, std::nullptr_t, T&& target_transform)
        : TransformDataset(
              dataset, nullptr,
              std::make_unique<std::remove_cvref_t<T>>(std::forward<T>(target_transform))) {}

    std::size_t size() const override;

    Sample get(std::size_t index) const override;

    std::pair<Tensor, Tensor> getBatch(std::span<const std::size_t> indices) const override;

private:
    const Dataset& dataset_;
    std::unique_ptr<transforms::Transform> input_transform_;
    std::unique_ptr<transforms::Transform> target_transform_;
};
} // namespace neuro
