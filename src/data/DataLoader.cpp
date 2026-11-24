#include "neuro/data/DataLoader.hpp"

#include "neuro/Tensor.hpp"
#include "neuro/data/Dataset.hpp"

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <random>
#include <span>
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
    std::span<std::size_t> batch_indices =
        std::span{loader_.sample_indices_}.subspan(position_, end - position_);

    return loader_.dataset_.getBatch(batch_indices);
};

DataLoader::Iterator& DataLoader::Iterator::operator++() {
    std::size_t end = std::min(position_ + loader_.batch_size_, loader_.dataset_.size());
    position_ = end;
    return *this;
};

bool DataLoader::Iterator::operator!=(Iterator other) const {
    return (other.position_ != position_) || (&loader_ != &other.loader_);
};
} // namespace neuro
