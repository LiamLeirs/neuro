#include "../include/neuro/data/DataLoader.hpp"

#include "neuro/Tensor.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace neuro {

DataLoader::DataLoader(Tensor input, Tensor target, int batch_size, bool shuffle)
    : input_(input), target_(target), batch_size_(batch_size), shuffle_(shuffle) {
    if (batch_size <= 0) {
        throw std::invalid_argument("Batch size must be greater than zero");
    }
    if (input_.shape()[0] != target_.shape()[0]) {
        throw std::invalid_argument("Input and targets must have same amount of samples");
    }
};

DataLoader::Iterator::Iterator(DataLoader& loader, int position)
    : loader_(loader), position_(position) {};

std::pair<Tensor, Tensor> DataLoader::Iterator::operator*() const {
    int end = std::min(position_ + loader_.batch_size_, (int)loader_.input_.shape()[0]);
    Tensor input_batch = loader_.input_.slice(position_, end);
    Tensor target_batch = loader_.target_.slice(position_, end);
    return std::pair<Tensor, Tensor>(input_batch, target_batch);
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
