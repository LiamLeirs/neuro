#pragma once

#include "neuro/Tensor.hpp"
#include "neuro/data/Dataset.hpp"

#include <vector>
namespace neuro {

class TensorDataset : public Dataset {
public:
    TensorDataset(const std::vector<Tensor>& inputs, const std::vector<Tensor>& targets);

    std::size_t size() const override;

    Sample get(std::size_t index) const override;

private:
    std::vector<Sample> samples_;
};
} // namespace neuro
