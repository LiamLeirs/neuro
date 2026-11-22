#include "neuro/data/DataLoader.hpp"

#include "neuro/Tensor.hpp"

#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace neuro {

DataLoader::DataLoader(Tensor input, Tensor target, int batch_size, bool shuffle, int seed)
    : input_(input), target_(target), batch_size_(batch_size), shuffle_(shuffle),
      rng_(std::mt19937(seed)) {
    if (batch_size <= 0) {
        throw std::invalid_argument("Batch size must be greater than zero");
    }
    if (input_.shape()[0] != target_.shape()[0]) {
        throw std::invalid_argument("Input and targets must have same amount of samples");
    }
    sample_indices_ = std::vector<int>(input.data().dims(0));
    std::iota(sample_indices_.begin(), sample_indices_.end(), 0);
};

DataLoader::Iterator::Iterator(DataLoader& loader, int position)
    : loader_(loader), position_(position) {};

std::pair<Tensor, Tensor> DataLoader::Iterator::operator*() const {
    int end = std::min(position_ + loader_.batch_size_, (int)loader_.input_.shape()[0]);
    if (!loader_.shuffle_) {
        Tensor input_batch = loader_.input_.slice(position_, end);
        Tensor target_batch = loader_.target_.slice(position_, end);
        return std::pair<Tensor, Tensor>(input_batch, target_batch);
    } else {
        std::vector<int> range(loader_.sample_indices_.begin() + position_,
                               loader_.sample_indices_.begin() + end);
        Tensor input_batch = loader_.input_.indexSelect(range);
        Tensor target_batch = loader_.target_.indexSelect(range);
        return std::pair<Tensor, Tensor>(input_batch, target_batch);
    }
}

DataLoader::Iterator& DataLoader::Iterator::operator++() {
    int end = std::min(position_ + loader_.batch_size_, (int)loader_.input_.shape()[0]);
    position_ = end;
    return *this;
};

bool DataLoader::Iterator::operator!=(Iterator other) const {
    return (other.position_ != position_) || (&loader_ != &other.loader_);
};
} // namespace neuro
