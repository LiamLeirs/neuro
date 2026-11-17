#pragma once

#include <af/array.h>

namespace neuro {

class Tensor {
public:
    explicit Tensor(const af::array& data);

private:
    af::array data_;
};

} // namespace neuro
