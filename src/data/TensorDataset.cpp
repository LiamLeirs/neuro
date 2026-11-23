#include "neuro/data/TensorDataset.hpp"

#include <cstddef>
#include <stdexcept>

namespace neuro {

TensorDataset::TensorDataset(const std::vector<Tensor>& inputs,
                             const std::vector<Tensor>& targets) {
    if (inputs.size() != targets.size())
        throw std::invalid_argument("Number of samples don't match");
    samples_.reserve(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); i++) {
        samples_.push_back(Sample{inputs[i], targets[i]});
    }
};

std::size_t TensorDataset::size() const {
    return samples_.size();
};

TensorDataset::Sample TensorDataset::get(std::size_t index) const {
    return samples_.at(index);
}
} // namespace neuro