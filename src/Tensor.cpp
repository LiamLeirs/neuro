#include "neuro/Tensor.hpp"

namespace neuro {

Tensor::Tensor(const af::array& data) : data_(data) {}

} // namespace neuro
