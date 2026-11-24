#pragma once

#include "neuro/Tensor.hpp"
#include "neuro/data/DataLoader.hpp"
#include "neuro/nn/Module.hpp"
#include "neuro/optim/Optimizer.hpp"

#include <cstddef>
#include <functional>

namespace neuro {

using LossFunction = std::function<Tensor(const Tensor&, const Tensor&)>;

class Trainer {
public:
    Trainer(nn::Module& model, Optimizer& optim, LossFunction criterion);

    void fit(DataLoader& loader, std::size_t num_epochs);

    double trainEpoch(DataLoader& loader);

private:
    nn::Module& model_;
    Optimizer& optim_;
    LossFunction criterion_;
};

} // namespace neuro