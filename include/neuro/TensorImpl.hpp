#pragma once

#include <af/array.h>
#include <functional>
#include <memory>
#include <vector>

namespace neuro {

struct TensorImpl {
    af::array data_;
    af::array grad_;
    bool requires_grad_ = false;

    std::vector<std::shared_ptr<TensorImpl>> dependencies;
    std::function<void()> backward_fn;
};

} // namespace neuro