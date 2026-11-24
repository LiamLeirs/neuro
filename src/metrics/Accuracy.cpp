#include "neuro/metrics/Accuracy.hpp"

#include <stdexcept>

namespace neuro::metrics {

void Accuracy::update(const Tensor& predictions, const Tensor& targets) {
    const auto pred_shape = predictions.shape();
    const auto target_shape = targets.shape();

    if (pred_shape[2] != 1 || pred_shape[3] != 1) {
        throw std::invalid_argument("Predictions must have shape (N, C)");
    }

    if (pred_shape[0] == 0 || pred_shape[1] == 0) {
        throw std::invalid_argument("Predictions must not be empty");
    }

    if (target_shape[1] != 1 || target_shape[2] != 1 || target_shape[3] != 1) {
        throw std::invalid_argument("Targets must have shape (N, 1)");
    }

    if (pred_shape[0] != target_shape[0]) {
        throw std::invalid_argument("Prediction and target batch sizes don't match");
    }

    if (!targets.data().isinteger()) {
        throw std::invalid_argument("Targets must contain integer class labels");
    }

    af::array out_of_bounds = (targets.data() < 0) || (targets.data() >= pred_shape[1]);

    if (af::anyTrue<bool>(out_of_bounds)) {
        throw std::invalid_argument("Target class index out of bounds");
    }

    Tensor predicted_classes = predictions.argmax(1);

    af::array correct = predicted_classes.data() == targets.data().as(u32);

    num_correct_ += af::sum<unsigned>(correct.as(u32));
    total_samples_ += pred_shape[0];
}

double Accuracy::compute() const {
    if (total_samples_ == 0)
        return 0.0;

    return static_cast<double>(num_correct_) / total_samples_;
}

void Accuracy::reset() {
    num_correct_ = 0;
    total_samples_ = 0;
}

} // namespace neuro::metrics