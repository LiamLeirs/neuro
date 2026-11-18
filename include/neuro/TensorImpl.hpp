#pragma once
#include <af/array.h>
namespace neuro {

struct TensorImpl {
    af::array data_;
    af::array grad_;
    bool requires_grad_;
};

} // namespace neuro
