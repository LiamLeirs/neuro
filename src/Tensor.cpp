#include "neuro/Tensor.hpp"

#include "neuro/GradMode.hpp"

#include <af/algorithm.h>
#include <af/arith.h>
#include <af/array.h>
#include <af/blas.h>
#include <af/data.h>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/gfor.h>
#include <af/image.h>
#include <af/index.h>
#include <af/ml.h>
#include <af/seq.h>
#include <af/signal.h>
#include <af/statistics.h>
#include <af/util.h>
#include <cmath>
#include <cstddef>
#include <memory>
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

Tensor Tensor::makeResult(const af::array& data, std::vector<std::shared_ptr<TensorImpl>> parents,
                          BackwardRule backward) {
    bool requires_grad = false;

    if (GradMode::isEnabled()) {
        for (const auto& parent : parents) {
            requires_grad |= parent->requires_grad_;
        }
    }

    Tensor result(data, requires_grad);

    if (!requires_grad)
        return result;

    result.impl_->dependencies = std::move(parents);

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

Tensor Tensor::reshape(const af::dim4& dims) const {
    if (dims.elements() != impl_->data_.elements()) {
        throw std::invalid_argument("Cannot reshape tensor: element counts do not match");
    }
    for (int i = 0; i < 4; ++i) {
        if (dims[i] <= 0) {
            throw std::invalid_argument("Reshape dimensions must be positive");
        }
    }
    af::dim4 original_shape = impl_->data_.dims();
    auto parent = impl_;
    af::array result = af::moddims(parent->data_, dims);
    return makeResult(result, {parent}, [parent, original_shape](const af::array& grad) {
        if (parent->requires_grad_) {
            parent->grad_ += af::moddims(grad, original_shape);
        }
    });
};

Tensor Tensor::reorder(int d0, int d1, int d2, int d3) const {
    const std::array<int, 4> perm = {d0, d1, d2, d3};
    std::array<int, 4> inversePerm{};
    std::array<bool, 4> seen{};

    for (int i = 0; i < 4; ++i) {
        const int dim = perm[i];

        if (dim < 0 || dim > 3) {
            throw std::invalid_argument("Dimensions must be between 0 and 3");
        }

        if (seen[dim]) {
            throw std::invalid_argument("Dimensions must form a valid permutation");
        }

        seen[dim] = true;
        inversePerm[dim] = i;
    }

    auto parent = impl_;
    af::array result = af::reorder(parent->data_, d0, d1, d2, d3);

    return makeResult(result, {parent}, [parent, inversePerm](const af::array& grad) {
        if (parent->requires_grad_) {
            parent->grad_ +=
                af::reorder(grad, inversePerm[0], inversePerm[1], inversePerm[2], inversePerm[3]);
        }
    });
}

Tensor Tensor::flatten() const {
    const auto dims = shape();

    const auto batch_size = dims[0];
    const auto feature_length = dims[1] * dims[2] * dims[3];

    return reshape(af::dim4(batch_size, feature_length));
}

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

Tensor Tensor::conv2d(const Tensor& weights, int stride, int padding, int dilation) const {
    // Validate convolution parameters.
    if (data().isempty() || weights.data().isempty()) {
        throw std::invalid_argument("Input and weights must not be empty");
    }
    if (stride < 1) {
        throw std::invalid_argument("Stride must be positive");
    }
    if (padding < 0) {
        throw std::invalid_argument("Padding must be non-negative");
    }
    if (dilation < 1) {
        throw std::invalid_argument("Dilation must be positive");
    }

    const auto input_shape = shape();
    const auto weight_shape = weights.shape();

    if (input_shape[1] != weight_shape[1]) {
        throw std::invalid_argument("Input and weight channel counts must match");
    }

    const auto effective_height = dilation * (weight_shape[2] - 1) + 1;
    const auto effective_width = dilation * (weight_shape[3] - 1) + 1;

    if (effective_height > input_shape[2] + 2LL * padding ||
        effective_width > input_shape[3] + 2LL * padding) {
        throw std::invalid_argument("Effective kernel exceeds padded input dimensions");
    }

    const af::dim4 stride_dims(stride, stride);
    const af::dim4 padding_dims(padding, padding);
    const af::dim4 dilation_dims(dilation, dilation);

    auto input_parent = impl_;
    auto weight_parent = weights.impl_;

    // Convert Neuro's NCHW layout to ArrayFire's WHCN layout.
    const af::array input = af::reorder(data(), 3, 2, 1, 0);
    af::array filters = af::reorder(weights.data(), 3, 2, 1, 0);

    // Reverse the spatial kernel dimensions to obtain cross-correlation.
    filters = af::flip(af::flip(filters, 0), 1);

    const af::array result =
        af::convolve2NN(input, filters, stride_dims, padding_dims, dilation_dims);

    const af::array output = af::reorder(result, 3, 2, 1, 0);

    return makeResult(output, {input_parent, weight_parent},
                      [input_parent, weight_parent, input, filters, result, stride_dims,
                       padding_dims, dilation_dims](const af::array& grad) {
                          const af::array reordered_grad = af::reorder(grad, 3, 2, 1, 0);

                          // Propagate gradients to the input.
                          if (input_parent->requires_grad_) {
                              const af::array input_grad = af::convolve2GradientNN(
                                  reordered_grad, input, filters, result, stride_dims, padding_dims,
                                  dilation_dims, AF_CONV_GRADIENT_DATA);

                              input_parent->grad_ += af::reorder(input_grad, 3, 2, 1, 0);
                          }

                          // Propagate gradients to the weights.
                          if (weight_parent->requires_grad_) {
                              af::array weight_grad = af::convolve2GradientNN(
                                  reordered_grad, input, filters, result, stride_dims, padding_dims,
                                  dilation_dims, AF_CONV_GRADIENT_FILTER);

                              // Undo the forward kernel reversal and layout conversion.
                              weight_grad = af::flip(af::flip(weight_grad, 0), 1);
                              weight_parent->grad_ += af::reorder(weight_grad, 3, 2, 1, 0);
                          }
                      });
}

Tensor Tensor::maxpool2d(int window_size, int stride, int padding) const {
    if (data().isempty())
        throw std::invalid_argument("Input must not be empty");

    if (data().type() != f32 && data().type() != f64)
        throw std::invalid_argument("Max pooling requires floating-point input");

    if (window_size < 1)
        throw std::invalid_argument("Window size must be positive");

    if (stride < 1)
        throw std::invalid_argument("Stride must be positive");

    if (padding < 0)
        throw std::invalid_argument("Padding must be non-negative");

    if (padding > window_size / 2)
        throw std::invalid_argument("Padding must not exceed half the window size");

    const auto dims = shape();
    const dim_t batch = dims[0];
    const dim_t channels = dims[1];
    const dim_t height = dims[2];
    const dim_t width = dims[3];

    const dim_t padded_h = height + 2LL * padding;
    const dim_t padded_w = width + 2LL * padding;

    if (window_size > padded_h || window_size > padded_w)
        throw std::invalid_argument("Window exceeds padded input dimensions");

    const dim_t out_h = (padded_h - window_size) / stride + 1;
    const dim_t out_w = (padded_w - window_size) / stride + 1;
    const dim_t window_elements = static_cast<dim_t>(window_size) * window_size;

    // Convert NCHW to ArrayFire's spatial layout (WHCN).
    const af::array input = af::reorder(data(), 3, 2, 1, 0);
    af::array padded = input;

    // Explicit negative-infinity padding prevents padded values
    // from becoming maxima when input values are negative.
    if (padding > 0) {
        padded = af::constant(-std::numeric_limits<double>::infinity(),
                              af::dim4(padded_w, padded_h, channels, batch), input.type());

        padded(af::seq(padding, padding + width - 1), af::seq(padding, padding + height - 1),
               af::span, af::span) = input;
    }

    // Extract pooling windows and retain their maximum indices.
    const af::array cols = af::unwrap(padded, window_size, window_size, stride, stride, 0, 0);

    af::array max_vals, max_indices;
    af::max(max_vals, max_indices, cols, 0);

    const af::array output =
        af::reorder(af::moddims(max_vals, out_w, out_h, channels, batch), 3, 2, 1, 0);

    auto parent = impl_;

    return makeResult(
        output, {parent},
        [parent, max_indices, channels, batch, width, height, padded_w, padded_h, out_w, out_h,
         window_elements, window_size, stride, padding](const af::array& grad) {
            if (!parent->requires_grad_)
                return;

            // Align upstream gradients with the pooling windows.
            const af::array grad_cols =
                af::moddims(af::reorder(grad, 3, 2, 1, 0), 1, out_w * out_h, channels, batch);

            // Construct a mask selecting one maximum per window.
            const af::array indices =
                af::tile(af::range(af::dim4(window_elements), 0, u32), max_indices.dims());

            const af::array mask = indices == af::tile(max_indices, window_elements);

            // Route each gradient to its selected maximum.
            const af::array window_grads =
                af::select(mask, af::tile(grad_cols, window_elements), 0.0);

            // Reconstruct spatial gradients, summing overlapping windows.
            const af::array padded_grad = af::wrap(window_grads, padded_w, padded_h, window_size,
                                                   window_size, stride, stride, 0, 0);

            af::array input_grad = padded_grad;

            // Remove explicit padding.
            if (padding > 0) {
                input_grad =
                    padded_grad(af::seq(padding, padding + width - 1),
                                af::seq(padding, padding + height - 1), af::span, af::span);
            }

            // Restore NCHW layout and accumulate gradients.
            parent->grad_ += af::reorder(input_grad, 3, 2, 1, 0);
        });
}

namespace {

af::array insert_dim(const af::array& in, int dim) {
    if (in.isempty())
        throw std::invalid_argument("Cannot insert dimension into an empty array");

    const af::dim4 old_dims = in.dims();

    // ArrayFire supports a maximum of four dimensions.
    if (old_dims[3] != 1)
        throw std::invalid_argument("Cannot insert dimension into a 4D tensor");

    // Determine the logical rank, ignoring trailing singleton dimensions.
    int rank = 1;
    for (int i = 2; i >= 0; --i) {
        if (old_dims[i] != 1) {
            rank = i + 1;
            break;
        }
    }

    if (dim < 0 || dim > rank)
        throw std::out_of_range("Invalid insertion dimension");

    af::dim4 new_dims(1, 1, 1, 1);

    for (int i = 0; i < dim; ++i)
        new_dims[i] = old_dims[i];

    for (int i = dim; i < rank; ++i)
        new_dims[i + 1] = old_dims[i];

    return af::moddims(in, new_dims);
}

} // namespace

Tensor Tensor::stack(const std::vector<Tensor>& tensors, int dim) {
    if (tensors.empty())
        throw std::invalid_argument("Cannot stack an empty collection");

    const af::dim4 expected_shape = tensors.front().shape();

    std::vector<std::shared_ptr<TensorImpl>> parents;
    parents.reserve(tensors.size());

    af::array output;

    for (const auto& tensor : tensors) {
        if (tensor.shape() != expected_shape)
            throw std::invalid_argument("All tensors must have the same shape");

        af::array sample = insert_dim(tensor.data(), dim);

        if (parents.empty())
            output = std::move(sample);
        else
            output = af::join(dim, output, sample);

        parents.push_back(tensor.impl_);
    }

    return makeResult(output, parents, [parents, dim](const af::array& grad) {
        for (std::size_t i = 0; i < parents.size(); ++i) {
            const auto& parent = parents[i];

            if (!parent->requires_grad_)
                continue;

            af::index indices[] = {af::span, af::span, af::span, af::span};

            indices[dim] = static_cast<int>(i);

            af::array grad_slice = grad(indices[0], indices[1], indices[2], indices[3]);

            grad_slice = af::moddims(grad_slice, parent->data_.dims());

            parent->grad_ += grad_slice;
        }
    });
}
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

Tensor Tensor::argmax(int dim) const {
    if (dim < 0 || dim >= 4) {
        throw std::out_of_range("Tensor dimension must be between 0 and 3");
    }
    af::array max_vals, max_idxs;
    af::max(max_vals, max_idxs, impl_->data_, dim);
    return Tensor(max_idxs);
};

Tensor Tensor::gather(const std::vector<int>& indices) const {
    af::array idx(indices.size(), indices.data());
    Tensor batch_indices(idx);
    return gather(batch_indices);
};

Tensor Tensor::gather(const Tensor& indices) const {
    if (impl_->data_.dims(2) != 1 || impl_->data_.dims(3) != 1)
        throw std::invalid_argument("Tensor must be 2D");
    if (indices.data().dims(0) != impl_->data_.dims(0) || indices.data().dims(1) != 1 ||
        indices.data().dims(2) != 1 || indices.data().dims(3) != 1)
        throw std::invalid_argument("Index dimensions don't match");
    if (!indices.data().isinteger()) {
        throw std::invalid_argument("Indices aren't integers");
    }
    auto parent = impl_;
    af::array outOfBounds = (indices.data() >= parent->data_.dims(1)) || (indices.data() < 0);
    if (af::anyTrue<bool>(outOfBounds)) {
        throw std::invalid_argument("Index out of bounds");
    }
    std::size_t batch_size = indices.data().dims(0);
    af::array rows = af::range(af::dim4(batch_size), 0, u32);
    af::array idx = rows + indices.data().as(u32) * batch_size;

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

Tensor Tensor::gather(std::initializer_list<int> indices) const {
    return gather(std::vector<int>(indices));
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

Tensor Tensor::leakyRelu(float negativeSlope) const {
    if (!std::isfinite(negativeSlope))
        throw std::invalid_argument("Negative slope must be finite");
    auto parent = impl_;
    const af::array& x = parent->data_;
    af::array result = af::select(x >= 0, x, negativeSlope * x);
    return makeResult(result, {parent}, [parent, negativeSlope](const af::array& grad) {
        if (parent->requires_grad_)
            parent->grad_ +=
                grad *
                af::select(parent->data_ >= 0,
                           af::constant(1, parent->data_.dims(), parent->data_.type()),
                           af::constant(negativeSlope, parent->data_.dims(), parent->data_.type()));
    });
}

Tensor Tensor::softmax(int dim) const {
    if (dim < 0 || dim > 3)
        throw std::invalid_argument("Dim out of range");
    Tensor shifted = *this - max(dim);
    Tensor exponentials = shifted.exp();
    Tensor normalizer = exponentials.sum(dim);
    return exponentials / normalizer;
};

Tensor Tensor::logSoftmax(int dim) const {
    if (dim < 0 || dim > 3)
        throw std::invalid_argument("Dim out of range");
    Tensor shifted = *this - max(dim);
    Tensor exponentials = shifted.exp();
    Tensor normalizer = exponentials.sum(dim);
    return shifted - normalizer.log();
};

} // namespace neuro