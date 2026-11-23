#include "neuro/data/DataLoader.hpp"

#include "neuro/Tensor.hpp"
#include "neuro/data/Dataset.hpp"

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace neuro {

DataLoader::DataLoader(const Dataset& dataset, int batch_size, bool shuffle, int seed)
    : dataset_(dataset), batch_size_(batch_size), shuffle_(shuffle), rng_(std::mt19937(seed)) {
    if (batch_size <= 0) {
        throw std::invalid_argument("Batch size must be greater than zero");
    }
    sample_indices_.resize(dataset_.size());
    std::iota(sample_indices_.begin(), sample_indices_.end(), 0);
};

DataLoader::Iterator::Iterator(DataLoader& loader, std::size_t position)
    : loader_(loader), position_(position) {};

std::pair<Tensor, Tensor> DataLoader::Iterator::operator*() const {
    const std::size_t end = std::min(position_ + static_cast<std::size_t>(loader_.batch_size_),
                                     loader_.dataset_.size());

    std::vector<Tensor> inputs;
    std::vector<Tensor> targets;

    inputs.reserve(end - position_);
    targets.reserve(end - position_);

    for (std::size_t i = position_; i < end; ++i) {
        const std::size_t sample_index = loader_.sample_indices_[i];

        Dataset::Sample sample = loader_.dataset_.get(sample_index);

        inputs.push_back(std::move(sample.input));
        targets.push_back(std::move(sample.target));
    }

    return {Tensor::stack(inputs, 0), Tensor::stack(targets, 0)};
}

DataLoader::Iterator& DataLoader::Iterator::operator++() {
    int end = std::min(position_ + loader_.batch_size_, loader_.dataset_.size());
    position_ = end;
    return *this;
};

bool DataLoader::Iterator::operator!=(Iterator other) const {
    return (other.position_ != position_) || (&loader_ != &other.loader_);
};
} // namespace neuro
