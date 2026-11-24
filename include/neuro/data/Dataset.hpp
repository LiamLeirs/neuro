#pragma once

#include "neuro/Tensor.hpp"

#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>
namespace neuro {

class Dataset {
public:
    struct Sample {
        Tensor input;
        Tensor target;
    };

    virtual ~Dataset() = default;

    virtual std::pair<Tensor, Tensor> getBatch(std::span<const std::size_t> indices) const {
        if (indices.empty())
            throw std::invalid_argument("Index vector is empty");
        std::vector<Tensor> inputs;
        inputs.reserve(indices.size());
        std::vector<Tensor> targets;
        targets.reserve(indices.size());
        for (auto index : indices) {
            Sample sample = get(index);
            inputs.push_back(std::move(sample.input));
            targets.push_back(std::move(sample.target));
        }
        Tensor input = Tensor::stack(inputs, 0);
        Tensor target = Tensor::stack(targets, 0);
        return {input, target};
    };

    virtual std::size_t size() const = 0;

    virtual Sample get(std::size_t index) const = 0;
};
} // namespace neuro
