#pragma once

#include <af/array.h>
#include <af/dim4.hpp>
#include <ostream>

namespace neuro {

class Tensor {
public:
    explicit Tensor(const af::array& data);
    const af::array& data() const;
    af::dim4 shape() const;
    Tensor operator+(const Tensor& other) const;

private:
    af::array data_;
};

std::ostream& operator<<(std::ostream& out, const Tensor& t);

} // namespace neuro
