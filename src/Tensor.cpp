#include "neuro/Tensor.hpp"

#include <af/algorithm.h>
#include <af/arith.h>
#include <af/array.h>
#include <af/blas.h>
#include <af/data.h>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/gfor.h>
#include <af/seq.h>
#include <af/statistics.h>
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

    return makeResult(af::batchFunc(left->data_, right->data_, af::operator+), {left, right},
                      [left, right](const af::array& grad) {
                          if (left->requires_grad_)
                              left->grad_ += sumToShape(grad, left->grad_.dims());

                          if (right->requires_grad_)
                              right->grad_ += sumToShape(grad, right->grad_.dims());
                      });
}

Tensor Tensor::operator-(const Tensor& other) const {
    auto left = impl_;
    auto right = other.impl_;

    return makeResult(af::batchFunc(left->data_, right->data_, af::operator-), {left, right},
                      [left, right](const af::array& grad) {
                          if (left->requires_grad_)
                              left->grad_ += sumToShape(grad, left->grad_.dims());

                          if (right->requires_grad_)
                              right->grad_ -= sumToShape(grad, right->grad_.dims());
                      });
}

Tensor Tensor::operator*(const Tensor& other) const {
    auto left = impl_;
    auto right = other.impl_;

    return makeResult(af::batchFunc(left->data_, right->data_, af::operator*), {left, right},
                      [left, right](const af::array& grad) {
                          if (left->requires_grad_)
                              left->grad_ +=
                                  sumToShape(af::batchFunc(grad, right->data_, af::operator*),
                                             left->data_.dims());

                          if (right->requires_grad_)
                              right->grad_ +=
                                  sumToShape(af::batchFunc(left->data_, grad, af::operator*),
                                             right->data_.dims());
                      });
}

Tensor Tensor::operator/(const Tensor& other) const {
    auto left = impl_;
    auto right = other.impl_;

    return makeResult(
        af::batchFunc(left->data_, right->data_, af::operator/), {left, right},
        [left, right](const af::array& grad) {
            if (left->requires_grad_)
                left->grad_ += sumToShape(af::batchFunc(grad, right->data_, af::operator/),
                                          left->data_.dims());

            if (right->requires_grad_)
                right->grad_ -= sumToShape(
                    af::batchFunc(af::batchFunc(grad, left->data_, af::operator*),
                                  af::batchFunc(right->data_, right->data_, af::operator*),
                                  af::operator/),
                    right->data_.dims());
        });
}

Tensor Tensor::abs() const {

    auto parent = impl_;

    return makeResult(af::abs(parent->data_), {parent}, [parent](const af::array& grad) {
        if (parent->requires_grad_)
            parent->grad_ += grad * ((parent->data_ > 0) - (parent->data_ < 0));
    });
};

Tensor Tensor::exp() const {
    auto parent = impl_;
    af::array result = af::exp(parent->data_);
    return makeResult(result, {parent}, [parent, result](const af::array& grad) {
        if (parent->requires_grad_)
            parent->grad_ += grad * result;
    });
};

Tensor Tensor::log() const {
    auto parent = impl_;
    return makeResult(af::log(parent->data_), {parent}, [parent](const af::array& grad) {
        if (parent->requires_grad_)
            parent->grad_ += grad / parent->data_;
    });
};

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

Tensor Tensor::operator[](int index) const {
    if (index < 0 || index >= impl_->data_.elements())
        throw std::out_of_range("Index out of bounds");
    if (!impl_->data_.isvector())
        throw std::invalid_argument("Tensor indexing only allowed for one dimensional tensors");
    return makeResult(impl_->data_(index), {impl_}, [parent = impl_, index](const af::array& grad) {
        if (parent->requires_grad_)
            parent->grad_(index) += grad;
    });
};

Tensor Tensor::slice(int start, int end) const {
    if (start >= end) {
        throw std::invalid_argument("Start must be smaller than end");
    }
    if (start < 0 || end > impl_->data_.dims(0))
        throw std::out_of_range("Index out of bounds");
    af::array t = impl_->data_.rows(start, end - 1);
    return makeResult(t, {impl_}, [parent = impl_, start, end](const af::array& grad) {
        if (parent->requires_grad_)
            parent->grad_.rows(start, end - 1) += grad;
    });
};

Tensor Tensor::indexSelect(const std::vector<int>& indices) const {
    if (indices.size() == 0)
        throw std::invalid_argument("Encountered empty indices vector");
    for (int i : indices) {
        if (i < 0)
            throw std::out_of_range("Index must be non-negative");
        if (i >= impl_->data_.dims(0)) {
            throw std::out_of_range("Index out of bounds");
        }
    }
    af::array idx(indices.size(), indices.data());
    af::array t = impl_->data_(idx, af::span);
    return makeResult(t, {impl_}, [parent = impl_, indices](const af::array& grad) {
        if (parent->requires_grad_) {
            int j = 0;
            for (int i : indices) {
                parent->grad_(i, af::span) += grad(j, af::span);
                j++;
            }
        }
    });
};

// ==================== Reductions ====================

Tensor Tensor::sum(int dim) const {
    if (dim < 0 || dim >= 4) {
        throw std::out_of_range("Tensor dimension must be between 0 and 3");
    }

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

Tensor Tensor::mean() const {
    auto parent = impl_;

    return makeResult(af::mean(af::flat(parent->data_)), {parent}, [parent](const af::array& grad) {
        if (parent->requires_grad_) {
            parent->grad_ += af::tile(grad / parent->data_.elements(), parent->data_.dims());
        }
    });
};

Tensor Tensor::mean(int dim) const {
    if (dim < 0 || dim >= 4) {
        throw std::out_of_range("Tensor dimension must be between 0 and 3");
    }

    auto parent = impl_;

    return makeResult(af::mean(parent->data_, dim), {parent}, [parent, dim](const af::array& grad) {
        if (!parent->requires_grad_)
            return;

        af::dim4 repeats(1, 1, 1, 1);
        repeats[dim] = parent->data_.dims(dim);

        parent->grad_ += af::tile(grad / parent->data_.dims(dim), repeats);
    });
};

Tensor Tensor::max(int dim) const {
    if (dim < 0 || dim >= 4) {
        throw std::out_of_range("Tensor dimension must be between 0 and 3");
    }

    auto parent = impl_;
    af::array result = af::max(parent->data_, dim);
    return makeResult(result, {parent}, [parent, dim, result](const af::array& grad) {
        if (!parent->requires_grad_)
            return;

        af::dim4 repeats(1, 1, 1, 1);
        repeats[dim] = parent->data_.dims(dim);

        af::array mask = parent->data_ == af::tile(result, repeats);
        af::array counts = af::sum(mask.as(f32), dim);

        parent->grad_ += mask.as(parent->data_.type()) * af::tile(grad / counts, repeats);
    });
};

Tensor Tensor::max() const {
    auto parent = impl_;
    af::array result = af::max(af::flat(parent->data_));

    return makeResult(result, {parent}, [parent, result](const af::array& grad) {
        if (!parent->requires_grad_)
            return;

        af::array mask = parent->data_ == result.scalar<float>();
        float counts = af::sum<float>(mask.as(f32));

        parent->grad_ += mask.as(parent->data_.type()) * (grad.scalar<float>() / counts);
    });
}

Tensor Tensor::gather(const std::vector<int>& indices) const {
    if (impl_->data_.dims(2) != 1 || impl_->data_.dims(3) != 1) {
        throw std::invalid_argument("Tensor must be 2D");
    }
    if (indices.size() != impl_->data_.dims(0)) {
        throw std::invalid_argument("Index dimensions don't match");
    }
    auto parent = impl_;
    std::vector<unsigned> linearIndices;
    linearIndices.reserve(indices.size());

    for (size_t i = 0; i < indices.size(); i++) {
        if (indices[i] < 0 || indices[i] >= parent->data_.dims(1))
            throw std::invalid_argument("Index out of range");

        linearIndices.push_back(i + indices[i] * parent->data_.dims(0));
    }

    af::array idx(linearIndices.size(), linearIndices.data());
    af::array flat = af::flat(parent->data_);
    af::array result = flat(idx);

    return makeResult(result, {parent}, [parent, idx](const af::array& grad) {
        if (!parent->requires_grad_)
            return;
        af::array scatter = af::flat(af::constant(0, parent->data_.dims(), parent->data_.type()));
        scatter(idx) = grad;
        parent->grad_ += af::moddims(scatter, parent->data_.dims());
    });
};

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

// ==================== Broadcasting ====================

af::array Tensor::sumToShape(const af::array& grad, const af::dim4& targetShape) {
    if (grad.dims() == targetShape)
        return grad;
    af::array result = grad;
    for (int dim = 0; dim < 4; ++dim) {
        int d = result.dims()[dim];
        int targetD = targetShape[dim];
        if (d == targetD)
            continue;
        else if (targetD == 1)
            result = af::sum(result, dim);
        else
            throw std::invalid_argument("Cannot broadcast array");
    }
    return result;
}

// ==================== Activation Functions ====================

Tensor Tensor::relu() const {
    auto parent = impl_;

    return makeResult(af::max(parent->data_, 0.0), {parent}, [parent](const af::array& grad) {
        if (parent->requires_grad_)
            parent->grad_ += grad * (parent->data_ > 0);
    });
}

Tensor Tensor::sigmoid() const {
    auto parent = impl_;
    const af::array& x = parent->data_;

    af::array expNegAbs = af::exp(-af::abs(x));

    af::array result = af::select(x >= 0, 1 / (1 + expNegAbs), expNegAbs / (1 + expNegAbs));

    return makeResult(result, {parent}, [parent, result](const af::array& grad) {
        if (parent->requires_grad_)
            parent->grad_ += grad * result * (1 - result);
    });
}

Tensor Tensor::tanh() const {
    auto parent = impl_;
    const af::array& x = parent->data_;
    af::array result = af::tanh(x);

    return makeResult(result, {parent}, [parent, result](const af::array& grad) {
        if (parent->requires_grad_)
            parent->grad_ += grad * (1 - (result * result));
    });
}

} // namespace neuro