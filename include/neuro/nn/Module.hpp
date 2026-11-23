#pragma once

#include "neuro/Tensor.hpp"

#include <vector>
namespace neuro::nn {

class Module {
public:
    virtual Tensor forward(const Tensor& input) const = 0;
    Tensor operator()(const Tensor& input) const {
        return forward(input);
    };
    virtual std::vector<Tensor> parameters() const {
        return {};
    };
    virtual ~Module() = default;
};

} // namespace neuro::nn