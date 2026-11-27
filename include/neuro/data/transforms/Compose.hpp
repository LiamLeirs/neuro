#pragma once

#include "neuro/data/transforms/Transform.hpp"

#include <concepts>
#include <vector>

namespace neuro::transforms {

class Compose : public Transform {
public:
    void add(std::unique_ptr<Transform> transform);
    Tensor forward(const Tensor& input) const override;
    template <typename... Transforms>
        requires((std::derived_from<std::decay_t<Transforms>, Transform> &&
                  std::is_rvalue_reference_v<Transforms&&> &&
                  !std::same_as<std::decay_t<Transforms>, Compose>) &&
                 ...)
    Compose(Transforms&&... transforms) {
        (add(std::make_unique<std::decay_t<Transforms>>(std::forward<Transforms>(transforms))),
         ...);
    };

    bool supportsBatch() const noexcept override;

    Tensor forwardBatch(const Tensor& input) const override;

private:
    std::vector<std::unique_ptr<Transform>> transforms_;
};

} // namespace neuro::transforms
