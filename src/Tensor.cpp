#include "neuro/Tensor.hpp"

namespace neuro {

Tensor::Tensor(const af::array& data) : data_(data) {}

const af::array& Tensor::data() const {
    return data_;
}

} // namespace neuro
