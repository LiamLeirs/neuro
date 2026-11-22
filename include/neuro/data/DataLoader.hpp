#pragma once
#include "neuro/Tensor.hpp"

#include <algorithm>
#include <random>
#include <utility>
#include <vector>
namespace neuro {

class DataLoader {
public:
    class Iterator {
    public:
        Iterator(DataLoader& loader, int position);
        std::pair<Tensor, Tensor> operator*() const;
        Iterator& operator++();
        bool operator!=(Iterator other) const;

    private:
        DataLoader& loader_;
        int position_;
    };
    DataLoader(Tensor input, Tensor target, int batch_size, bool shuffle = false, int seed = 42);
    auto begin() {
        if (shuffle_) {
            std::shuffle(sample_indices_.begin(), sample_indices_.end(), rng_);
        }
        return Iterator(*this, 0);
    };
    auto end() {
        return Iterator(*this, input_.shape()[0]);
    }

private:
    std::mt19937 rng_;
    std::vector<int> sample_indices_;
    Tensor input_;
    Tensor target_;
    int batch_size_;
    bool shuffle_;
};

} // namespace neuro
