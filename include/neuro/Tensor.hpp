#pragma once

#include <af/array.h>
#include <af/dim4.hpp>
#include <initializer_list>
#include <ostream>

namespace neuro {

class Tensor {
public:
    explicit Tensor(const af::array& data);
    Tensor(std::initializer_list<float> value);
    Tensor(std::initializer_list<std::initializer_list<float>> value);
    const af::array& data() const;
    af::dim4 shape() const;
    Tensor operator+(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor matmul(const Tensor& other) const;
    Tensor sum(int dim) const;
    Tensor sum() const;
    Tensor transpose() const;

private:
    af::array data_;
};

std::ostream& operator<<(std::ostream& out, const Tensor& t);
Tensor operator*(const Tensor& t, double scalar);
Tensor operator*(double scalar, const Tensor& t);

} // namespace neuro
