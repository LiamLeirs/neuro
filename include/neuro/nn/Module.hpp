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
    virtual void train(bool mode = true) {
        training_ = mode;
    };
    void eval() {
        train(false);
    };
    bool isTraining() const noexcept {
        return training_;
    };

private:
    bool training_ = true;
};

} // namespace neuro::nn