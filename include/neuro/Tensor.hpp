#pragma once

#include "neuro/TensorImpl.hpp"
#include "unordered_set"

#include <af/array.h>
#include <af/dim4.hpp>
#include <initializer_list>
#include <memory>
#include <ostream>
#include <stack>

namespace neuro {

class Tensor {
public:
    explicit Tensor(const af::array& data, bool requires_grad = false);
    Tensor(std::initializer_list<float> value, bool requires_grad = false);
    Tensor(std::initializer_list<std::initializer_list<float>> value, bool requires_grad = false);
    const af::array& data() const;
    const af::array& grad() const;
    af::dim4 shape() const;
    Tensor operator+(const Tensor& other) const;
    Tensor operator+(double scalar) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator-(double scalar) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator*(double scalar) const;
    Tensor operator/(const Tensor& other) const;
    Tensor operator/(double scalar) const;
    Tensor matmul(const Tensor& other) const;
    Tensor sum(int dim) const;
    Tensor sum() const;
    Tensor transpose() const;
    bool requires_grad() const;

private:
    std::shared_ptr<TensorImpl> impl_;
    std::vector<std::shared_ptr<TensorImpl>> topologicalSort() const;
    void findTopoSort(const std::shared_ptr<TensorImpl>& node,
                      std::unordered_set<TensorImpl*>& visited,
                      std::vector<std::shared_ptr<TensorImpl>>& topo) const;
};

std::ostream& operator<<(std::ostream& out, const Tensor& t);
Tensor operator*(double scalar, const Tensor& t);
Tensor operator+(double scalar, const Tensor& t);
Tensor operator-(double scalar, const Tensor& t);
Tensor operator/(double scalar, const Tensor& t);
} // namespace neuro
