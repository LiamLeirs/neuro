#pragma once

#include "neuro/nn/Module.hpp"

#include <concepts>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>
namespace neuro::nn {

class Sequential : public Module {
public:
    void add(std::unique_ptr<Module> module);
    Tensor forward(const Tensor& input) const override;
    std::vector<Tensor> parameters() const override;
    template <typename... Modules>
        requires((std::derived_from<std::decay_t<Modules>, Module> &&
                  std::is_rvalue_reference_v<Modules&&> &&
                  !std::same_as<std::decay_t<Modules>, Sequential>) &&
                 ...)
    Sequential(Modules&&... modules) {
        (add(std::make_unique<std::decay_t<Modules>>(std::forward<Modules>(modules))), ...);
    };
    void train(bool mode = true) override;

private:
    std::vector<std::unique_ptr<Module>> modules_;
};

} // namespace neuro::nn
