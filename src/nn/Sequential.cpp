#include "neuro/nn/Sequential.hpp"

#include <stdexcept>
#include <vector>

namespace neuro::nn {
void Sequential::add(std::unique_ptr<Module> module) {
    if (module == nullptr) {
        throw std::invalid_argument("Module can't be null");
    }
    module->train(isTraining());
    modules_.push_back(std::move(module));
}

Tensor Sequential::forward(const Tensor& input) const {
    Tensor output = input;
    for (const auto& module : modules_) {
        output = module->forward(output);
    }
    return output;
};

std::vector<Tensor> Sequential::parameters() const {
    std::vector<Tensor> params;
    for (const auto& module : modules_) {
        for (auto param : module->parameters()) {
            params.push_back(param);
        }
    }
    return params;
};

void Sequential::train(bool mode) {
    Module::train(mode);
    for (auto& child : modules_) {
        child->train(mode);
    }
};
} // namespace neuro::nn
