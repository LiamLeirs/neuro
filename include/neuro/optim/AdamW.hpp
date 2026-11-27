#pragma once
#include "neuro/Tensor.hpp"
#include "neuro/optim/AdamBase.hpp"

#include <vector>

namespace neuro {

class AdamW : public AdamBase {
public:
    AdamW(std::vector<Tensor> parameters, double lr, double b1 = 0.9, double b2 = 0.999,
          double eps = 1e-8, double weight_decay = 0.01);
    void step() override;
};

} // namespace neuro
