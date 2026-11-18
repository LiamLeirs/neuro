#include "neuro/Tensor.hpp"

#include <af/algorithm.h>
#include <af/arith.h>
#include <af/blas.h>
#include <af/data.h>
#include <af/util.h>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace neuro {

namespace {

// Convert row-major initializer lists into an ArrayFire matrix.
af::array makeMatrix(std::initializer_list<std::initializer_list<float>> rows) {
    if (rows.size() == 0)
        return af::array();

    const std::size_t columns = rows.begin()->size();

    std::vector<float> values;
    values.reserve(rows.size() * columns);

    for (const auto& row : rows) {
        if (row.size() == 0)
            throw std::invalid_argument("Encountered empty row");

        if (row.size() != columns)
            throw std::invalid_argument("Wrong tensor dimensions");

        values.insert(values.end(), row.begin(), row.end());
    }

    return af::array(columns, rows.size(), values.data(), afHost).T();
}

// Automatically free ArrayFire's host-allocated printing buffer.
struct HostString {
    char* data = nullptr;

    ~HostString() {
        if (data != nullptr)
            af::freeHost(data);
    }
};

} // namespace

// ==================== Constructors ====================

Tensor::Tensor(const af::array& data, bool requires_grad) : impl_(std::make_shared<TensorImpl>()) {

    impl_->data_ = data;

    impl_->grad_ = data.elements() == 0 ? af::array() : af::constant(0, data.dims(), data.type());

    impl_->requires_grad_ = requires_grad;
}

Tensor::Tensor(std::initializer_list<float> value, bool requires_grad)
    : Tensor(af::array(value), requires_grad) {}

Tensor::Tensor(std::initializer_list<std::initializer_list<float>> value, bool requires_grad)
    : Tensor(makeMatrix(value), requires_grad) {}

// ==================== Getters ====================

const af::array& Tensor::data() const {
    return impl_->data_;
}

const af::array& Tensor::grad() const {
    return impl_->grad_;
}

af::dim4 Tensor::shape() const {
    return impl_->data_.dims();
}

bool Tensor::requires_grad() const {
    return impl_->requires_grad_;
}

// ==================== Autograd Setup ====================

Tensor Tensor::makeResult(const af::array& data,
                          std::initializer_list<std::shared_ptr<TensorImpl>> parents,
                          BackwardRule backward) {
    bool requires_grad = false;

    for (const auto& parent : parents) {
        requires_grad |= parent->requires_grad_;
    }

    Tensor result(data, requires_grad);

    if (!requires_grad)
        return result;

    result.impl_->dependencies.assign(parents.begin(), parents.end());

    std::weak_ptr<TensorImpl> output = result.impl_;

    result.impl_->backward_fn = [output, rule = std::move(backward)]() {
        if (auto out = output.lock()) {
            rule(out->grad_);
        }
    };

    return result;
}

// ==================== Tensor Arithmetic ====================

Tensor Tensor::operator+(const Tensor& other) const {
    auto left = impl_;
    auto right = other.impl_;

    return makeResult(left->data_ + right->data_, {left, right},
                      [left, right](const af::array& grad) {
                          if (left->requires_grad_)
                              left->grad_ += grad;

                          if (right->requires_grad_)
                              right->grad_ += grad;
                      });
}

Tensor Tensor::operator-(const Tensor& other) const {
    auto left = impl_;
    auto right = other.impl_;

    return makeResult(left->data_ - right->data_, {left, right},
                      [left, right](const af::array& grad) {
                          if (left->requires_grad_)
                              left->grad_ += grad;

                          if (right->requires_grad_)
                              right->grad_ -= grad;
                      });
}

Tensor Tensor::operator*(const Tensor& other) const {
    auto left = impl_;
    auto right = other.impl_;

    return makeResult(left->data_ * right->data_, {left, right},
                      [left, right](const af::array& grad) {
                          if (left->requires_grad_)
                              left->grad_ += grad * right->data_;

                          if (right->requires_grad_)
                              right->grad_ += grad * left->data_;
                      });
}

Tensor Tensor::operator/(const Tensor& other) const {
    auto left = impl_;
    auto right = other.impl_;

    return makeResult(left->data_ / right->data_, {left, right},
                      [left, right](const af::array& grad) {
                          if (left->requires_grad_)
                              left->grad_ += grad / right->data_;

                          if (right->requires_grad_)
                              right->grad_ -= grad * left->data_ / (right->data_ * right->data_);
                      });
}

// ==================== Scalar Arithmetic ====================

Tensor Tensor::operator+(double scalar) const {
    auto parent = impl_;

    return makeResult(parent->data_ + scalar, {parent}, [parent](const af::array& grad) {
        if (parent->requires_grad_)
            parent->grad_ += grad;
    });
}

Tensor Tensor::operator-(double scalar) const {
    return *this + (-scalar);
}

Tensor Tensor::operator*(double scalar) const {
    auto parent = impl_;

    return makeResult(parent->data_ * scalar, {parent}, [parent, scalar](const af::array& grad) {
        if (parent->requires_grad_)
            parent->grad_ += grad * scalar;
    });
}

Tensor Tensor::operator/(double scalar) const {
    auto parent = impl_;

    return makeResult(parent->data_ / scalar, {parent}, [parent, scalar](const af::array& grad) {
        if (parent->requires_grad_)
            parent->grad_ += grad / scalar;
    });
}

// ==================== Non-member Operators ====================

Tensor operator+(double scalar, const Tensor& t) {
    return t + scalar;
}

Tensor operator-(double scalar, const Tensor& t) {
    return -1.0 * t + scalar;
}

Tensor operator*(double scalar, const Tensor& t) {
    return t * scalar;
}

Tensor operator/(double scalar, const Tensor& t) {
    af::array numerator = af::constant(scalar, t.shape(), t.data().type());

    return Tensor(numerator) / t;
}

// ==================== Matrix Operations ====================

Tensor Tensor::matmul(const Tensor& other) const {
    auto left = impl_;
    auto right = other.impl_;

    return makeResult(af::matmul(left->data_, right->data_), {left, right},
                      [left, right](const af::array& grad) {
                          if (left->requires_grad_) {
                              left->grad_ += af::matmul(grad, af::transpose(right->data_));
                          }

                          if (right->requires_grad_) {
                              right->grad_ += af::matmul(af::transpose(left->data_), grad);
                          }
                      });
}

// ==================== Reductions ====================

Tensor Tensor::sum(int dim) const {
    auto parent = impl_;

    return makeResult(af::sum(parent->data_, dim), {parent}, [parent, dim](const af::array& grad) {
        if (!parent->requires_grad_)
            return;

        af::dim4 repeats(1, 1, 1, 1);
        repeats[dim] = parent->data_.dims(dim);

        parent->grad_ += af::tile(grad, repeats);
    });
}

Tensor Tensor::sum() const {
    auto parent = impl_;

    return makeResult(af::sum(af::flat(parent->data_)), {parent}, [parent](const af::array& grad) {
        if (parent->requires_grad_) {
            parent->grad_ += af::tile(grad, parent->data_.dims());
        }
    });
}

// ==================== Transpose ====================

Tensor Tensor::transpose() const {
    auto parent = impl_;

    return makeResult(parent->data_.T(), {parent}, [parent](const af::array& grad) {
        if (parent->requires_grad_)
            parent->grad_ += grad.T();
    });
}

// ==================== Topological Sorting ====================

std::vector<std::shared_ptr<TensorImpl>> Tensor::topologicalSort() const {
    std::unordered_set<TensorImpl*> visited;
    std::vector<std::shared_ptr<TensorImpl>> topo;

    findTopoSort(impl_, visited, topo);

    return topo;
}

void Tensor::findTopoSort(const std::shared_ptr<TensorImpl>& node,
                          std::unordered_set<TensorImpl*>& visited,
                          std::vector<std::shared_ptr<TensorImpl>>& topo) const {
    visited.insert(node.get());

    for (const auto& dependency : node->dependencies) {
        if (!visited.contains(dependency.get())) {
            findTopoSort(dependency, visited, topo);
        }
    }

    topo.push_back(node);
}

// ==================== Backpropagation ====================

void Tensor::backward() {
    if (!impl_->requires_grad_)
        return;

    if (impl_->data_.elements() != 1)
        throw std::invalid_argument("backward() requires a scalar output");

    impl_->grad_ = af::constant(1.0, impl_->data_.dims(), impl_->data_.type());

    const auto nodes = topologicalSort();

    for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
        if ((*it)->backward_fn)
            (*it)->backward_fn();
    }
}

void Tensor::zeroGrad() {
    impl_->grad_ = af::constant(0, impl_->data_.dims(), impl_->data_.type());
}

// ==================== Stream Output ====================

std::ostream& operator<<(std::ostream& out, const Tensor& t) {
    HostString str;

    af::toString(&str.data, "", t.data(), 4, true);

    if (str.data != nullptr)
        out << str.data;

    return out;
}

} // namespace neuro