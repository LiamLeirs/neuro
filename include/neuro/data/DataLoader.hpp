#pragma once
#include "neuro/Tensor.hpp"
#include "neuro/data/Dataset.hpp"

#include <algorithm>
#include <cstddef>
#include <random>
#include <utility>
#include <vector>
namespace neuro {

class DataLoader {
public:
    class Iterator {
    public:
        Iterator(DataLoader& loader, std::size_t position);
        std::pair<Tensor, Tensor> operator*() const;
        Iterator& operator++();
        bool operator!=(Iterator other) const;

    private:
        DataLoader& loader_;
        std::size_t position_;
    };
    DataLoader(const Dataset& dataset, int batch_size, bool shuffle = false, int seed = 42);
    auto begin() {
        if (shuffle_) {
            std::shuffle(sample_indices_.begin(), sample_indices_.end(), rng_);
        }
        return Iterator(*this, 0);
    };
    auto end() {
        return Iterator(*this, dataset_.size());
    }

private:
    std::mt19937 rng_;
    std::vector<std::size_t> sample_indices_;
    const Dataset& dataset_;
    int batch_size_;
    bool shuffle_;
};

} // namespace neuro
