#pragma once
#include "neuro/Tensor.hpp"

#include <utility>
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
    DataLoader(Tensor input, Tensor target, int batch_size, bool shuffle = false);
    auto begin() {
        return Iterator(*this, 0);
    };
    auto end() {
        return Iterator(*this, input_.shape()[0]);
    }

private:
    Tensor input_;
    Tensor target_;
    int batch_size_;
    bool shuffle_;
};

} // namespace neuro
