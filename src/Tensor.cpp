#include "neuro/Tensor.hpp"

#include <af/algorithm.h>
#include <af/array.h>
#include <af/blas.h>
#include <af/data.h>
#include <af/defines.h>
#include <af/util.h>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <vector>

namespace neuro {

Tensor::Tensor(const af::array& data, bool requires_grad)
    : data_(data), grad_(af::constant(0, data.dims(), data_.type())),
      requires_grad_(requires_grad) {}

Tensor::Tensor(std::initializer_list<float> value, bool requires_grad)
    : data_(af::array(value)), grad_(af::constant(0, data_.dims())), requires_grad_(requires_grad) {
}

Tensor::Tensor(std::initializer_list<std::initializer_list<float>> value, bool requires_grad)
    : requires_grad_(requires_grad) {
    if (value.size() == 0) {
        data_ = af::array();
        grad_ = af::array();
        return;
    }
    std::size_t rowSize = value.begin()->size();
    std::vector<float> array = {};
    for (const auto& row : value) {
        if (row.size() == 0)
            throw std::invalid_argument("Encountered empty row");
        if (row.size() != rowSize)
            throw std::invalid_argument("Wrong tensor dimensions");
        for (float scalar : row) {
            array.push_back(scalar);
        }
    }
    data_ = af::array(rowSize, value.size(), array.data(), afHost).T();
    grad_ = af::constant(0, data_.dims());
}

const af::array& Tensor::data() const {
    return data_;
}

const af::array& Tensor::grad() const {
    return grad_;
}

bool Tensor::requires_grad() const {
    return requires_grad_;
}

af::dim4 Tensor::shape() const {
    return data_.dims();
}

Tensor Tensor::operator+(const Tensor& other) const {
    return Tensor(data_ + other.data_);
}

Tensor Tensor::operator*(const Tensor& other) const {
    return Tensor(data_ * other.data_);
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

Tensor operator*(const Tensor& t, double scalar) {
    return Tensor(t.data() * scalar);
}

Tensor operator*(double scalar, const Tensor& t) {
    return t * scalar;
}

Tensor Tensor::matmul(const Tensor& other) const {
    return Tensor(af::matmul(data_, other.data_));
}

Tensor Tensor::sum(int dim) const {
    return Tensor(af::sum(data_, dim));
}

Tensor Tensor::sum() const {
    return Tensor(af::sum(af::flat(data_)));
}

Tensor Tensor::transpose() const {
    return Tensor(data_.T());
}

} // namespace neuro
