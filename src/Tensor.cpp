#include "neuro/Tensor.hpp"

#include "neuro/TensorImpl.hpp"

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

Tensor::Tensor(const af::array& data, bool requires_grad) : impl_(std::make_shared<TensorImpl>()) {
    impl_->data_ = data;
    impl_->grad_ = af::constant(0, data.dims(), impl_->data_.type());
    impl_->requires_grad_ = requires_grad;
}

Tensor::Tensor(std::initializer_list<float> value, bool requires_grad)
    : impl_(std::make_shared<TensorImpl>()) {
    impl_->data_ = af::array(value);
    impl_->grad_ = af::constant(0, impl_->data_.dims());
    impl_->requires_grad_ = requires_grad;
}

Tensor::Tensor(std::initializer_list<std::initializer_list<float>> value, bool requires_grad)
    : impl_(std::make_shared<TensorImpl>()) {
    impl_->requires_grad_ = requires_grad;
    if (value.size() == 0) {
        impl_->data_ = af::array();
        impl_->grad_ = af::array();
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
    impl_->data_ = af::array(rowSize, value.size(), array.data(), afHost).T();
    impl_->grad_ = af::constant(0, impl_->data_.dims());
}

const af::array& Tensor::data() const {
    return impl_->data_;
}

const af::array& Tensor::grad() const {
    return impl_->grad_;
}

bool Tensor::requires_grad() const {
    return impl_->requires_grad_;
}

af::dim4 Tensor::shape() const {
    return impl_->data_.dims();
}

Tensor Tensor::operator+(const Tensor& other) const {
    return Tensor(impl_->data_ + other.impl_->data_,
                  other.impl_->requires_grad_ || impl_->requires_grad_);
}

Tensor Tensor::operator*(const Tensor& other) const {
    return Tensor(impl_->data_ * other.impl_->data_,
                  other.impl_->requires_grad_ || impl_->requires_grad_);
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
    return Tensor(t.data() * scalar, t.requires_grad());
}

Tensor operator*(double scalar, const Tensor& t) {
    return t * scalar;
}

Tensor Tensor::matmul(const Tensor& other) const {
    return Tensor(af::matmul(impl_->data_, other.impl_->data_),
                  other.impl_->requires_grad_ || impl_->requires_grad_);
}

Tensor Tensor::sum(int dim) const {
    return Tensor(af::sum(impl_->data_, dim), impl_->requires_grad_);
}

Tensor Tensor::sum() const {
    return Tensor(af::sum(af::flat(impl_->data_)), impl_->requires_grad_);
}

Tensor Tensor::transpose() const {
    return Tensor(impl_->data_.T(), impl_->requires_grad_);
}

} // namespace neuro
