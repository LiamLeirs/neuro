#include "neuro/data/transforms/Compose.hpp"

#include <stdexcept>
#include <vector>

namespace neuro::transforms {
void Compose::add(std::unique_ptr<Transform> transform) {
    if (transform == nullptr) {
        throw std::invalid_argument("Transform can't be null");
    }
    transforms_.push_back(std::move(transform));
}

Tensor Compose::forward(const Tensor& input) const {
    Tensor output = input;
    for (const auto& transform : transforms_) {
        output = transform->forward(output);
    }
    return output;
};
} // namespace neuro::transforms
