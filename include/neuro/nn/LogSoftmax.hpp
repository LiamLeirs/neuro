#pragma once
#include "neuro/Tensor.hpp"
#include "neuro/nn/Module.hpp"

#include <stdexcept>
namespace neuro::nn {

class LogSoftmax : public Module {
public:
    LogSoftmax(int dim) : dim_(dim) {
        if (dim < 0 || dim > 3)
            throw std::invalid_argument("Dim out of range");
    }
    Tensor forward(const Tensor& input) const override {
        return input.logSoftmax(dim_);
    }

private:
    int dim_;
};

} // namespace neuro::nn