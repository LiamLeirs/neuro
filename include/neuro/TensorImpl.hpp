#pragma once
#include <af/array.h>
#include <memory>
#include <vector>
namespace neuro {

struct TensorImpl {
    af::array data_;
    af::array grad_;
    bool requires_grad_;
    std::vector<std::shared_ptr<TensorImpl>> dependencies;
};

} // namespace neuro
