#include "neuro/Tensor.hpp"

#include <af/util.h>
#include <iostream>

namespace neuro {

Tensor::Tensor(const af::array& data) : data_(data) {}

const af::array& Tensor::data() const {
    return data_;
}

af::dim4 Tensor::shape() const {
    return data_.dims();
}

Tensor Tensor::operator+(const Tensor& other) const {
    return Tensor(data_ + other.data_);
}

std::ostream& operator<<(std::ostream& out, const Tensor& t) {
    char* str = nullptr;
    af::toString(&str, "", t.data(), 4, true);

    if (str != nullptr) {
        out << str;
        af::freeHost(str); // Prevent host memory leak
    }
    return out;
}
} // namespace neuro
