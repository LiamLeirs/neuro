#pragma once

#include <af/array.h>

namespace neuro {

class Tensor {
public:
    explicit Tensor(const af::array& data);
    const af::array& data() const;

private:
    af::array data_;
};

} // namespace neuro
