#pragma once

#include "neuro/TensorImpl.hpp"

#include <af/array.h>
#include <af/dim4.hpp>
#include <functional>
#include <initializer_list>
#include <memory>
#include <ostream>
#include <unordered_set>
#include <vector>

namespace neuro {

class Tensor {
public:
    explicit Tensor(const af::array& data, bool requires_grad = false);
    Tensor();
    Tensor(std::initializer_list<float> value, bool requires_grad = false);
    Tensor(std::initializer_list<std::initializer_list<float>> value, bool requires_grad = false);

    const af::array& data() const;
    const af::array& grad() const;
    af::dim4 shape() const;
    bool requires_grad() const;

    static Tensor stack(const std::vector<Tensor>& tensors, int dim = 0);

    Tensor operator+(const Tensor& other) const;
    Tensor operator+(double scalar) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator-(double scalar) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator*(double scalar) const;
    Tensor operator/(const Tensor& other) const;
    Tensor operator/(double scalar) const;
    Tensor operator[](int index) const;

    Tensor pad(const af::dim4& lower, const af::dim4& upper) const;
    Tensor flatten() const;
    Tensor reorder(int d0, int d1, int d2, int d3) const;
    Tensor flip(int dim) const;
    Tensor reshape(const af::dim4& dims) const;
    Tensor matmul(const Tensor& other) const;
    Tensor sum(int dim) const;
    Tensor sum() const;
    Tensor mean(int dim) const;
    Tensor mean() const;
    Tensor max(int dim) const;
    Tensor max() const;
    Tensor gather(const std::vector<int>& indices) const;
    Tensor gather(const Tensor& indices) const;
    Tensor gather(std::initializer_list<int> indices) const;
    Tensor argmax(int dim) const;
    Tensor abs() const;
    Tensor exp() const;
    Tensor log() const;
    Tensor transpose() const;
    Tensor slice(int start, int end) const;
    Tensor slice(const af::dim4& start, const af::dim4& end) const;
    Tensor indexSelect(const std::vector<int>& indices) const;
    Tensor var() const;
    Tensor var(int dim) const;
    Tensor sqrt() const;
    Tensor crop(const af::dim4& start, const af::dim4& size) const;

    Tensor conv2d(const Tensor& weights, int stride = 1, int padding = 0, int dilation = 1) const;
    Tensor maxpool2d(int window_size, int stride = 1, int padding = 1) const;
    Tensor avgpool2d(int window_size, int stride, int padding, bool count_include_pad = true) const;
    Tensor globalAvgPool2d() const;

    Tensor relu() const;
    Tensor sigmoid() const;
    Tensor tanh() const;
    Tensor leakyRelu(float negativeSlope = 0.01f) const;
    Tensor softmax(int dim) const;
    Tensor logSoftmax(int dim) const;
    Tensor dropout(double p = 0.5) const;

    static Tensor where(const Tensor& condition, const Tensor& x, const Tensor& y);

    void backward();
    void zeroGrad();

private:
    using BackwardRule = std::function<void(const af::array&)>;

    std::shared_ptr<TensorImpl> impl_;

    // Registers dependencies and the operation's backward rule.
    static Tensor makeResult(const af::array& data,
                             std::vector<std::shared_ptr<TensorImpl>> parents,
                             BackwardRule backward);

    std::vector<std::shared_ptr<TensorImpl>> topologicalSort() const;

    void findTopoSort(const std::shared_ptr<TensorImpl>& node,
                      std::unordered_set<TensorImpl*>& visited,
                      std::vector<std::shared_ptr<TensorImpl>>& topo) const;

    static af::array sumToShape(const af::array& grad, const af::dim4& targetShape);

    friend class Optimizer;
};

std::ostream& operator<<(std::ostream& out, const Tensor& t);

Tensor operator+(double scalar, const Tensor& t);
Tensor operator-(double scalar, const Tensor& t);
Tensor operator*(double scalar, const Tensor& t);
Tensor operator/(double scalar, const Tensor& t);

} // namespace neuro