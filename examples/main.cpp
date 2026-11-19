
#include "neuro/Tensor.hpp"

#include <arrayfire.h>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace neuro;

// ==================== Test Utilities ====================

void checkArray(const af::array& actual, const af::dim4& expectedShape,
                const std::vector<float>& expected, const std::string& label) {

    if (actual.dims() != expectedShape)
        throw std::runtime_error(label + ": incorrect shape");

    if (actual.elements() != static_cast<dim_t>(expected.size()))
        throw std::runtime_error(label + ": incorrect element count");

    std::vector<float> values(expected.size());
    actual.host(values.data());

    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::abs(values[i] - expected[i]) > 1e-5f) {
            throw std::runtime_error(label + ": mismatch at index " + std::to_string(i) +
                                     " (expected " + std::to_string(expected[i]) + ", got " +
                                     std::to_string(values[i]) + ")");
        }
    }
}

template <typename Exception>
void expectThrow(const std::function<void()>& fn, const std::string& label) {
    try {
        fn();
    } catch (const Exception&) {
        return;
    }

    throw std::runtime_error(label + ": expected exception not thrown");
}

// ==================== Forward Tests ====================

void testForwardSlicing() {
    Tensor x(af::iota(af::dim4(4, 3), af::dim4(1), f32), true);

    Tensor y = x.slice(1, 3);

    // ArrayFire uses column-major ordering.
    checkArray(y.data(), af::dim4(2, 3), {1, 2, 5, 6, 9, 10}, "Forward slicing");

    std::cout << "[PASS] Forward slicing\n";
}

void testFullSlice() {
    Tensor x(af::iota(af::dim4(4, 3), af::dim4(1), f32), true);

    Tensor y = x.slice(0, 4);

    checkArray(y.data(), af::dim4(4, 3), {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, "Full slice");

    std::cout << "[PASS] Full slice\n";
}

void testSingleRow() {
    Tensor x(af::iota(af::dim4(4, 3), af::dim4(1), f32), true);

    Tensor y = x.slice(2, 3);

    checkArray(y.data(), af::dim4(1, 3), {2, 6, 10}, "Single row slice");

    std::cout << "[PASS] Single row slicing\n";
}

void testLastRow() {
    Tensor x(af::iota(af::dim4(4, 3), af::dim4(1), f32), true);

    Tensor y = x.slice(3, 4);

    checkArray(y.data(), af::dim4(1, 3), {3, 7, 11}, "Last row slice");

    std::cout << "[PASS] Last row slicing\n";
}

// ==================== Backward Tests ====================

void testBackwardPropagation() {
    Tensor x(af::iota(af::dim4(4, 3), af::dim4(1), f32), true);

    Tensor y = x.slice(1, 3);
    Tensor loss = (y * y).sum();

    loss.backward();

    // x =
    // [ 0  4  8 ]
    // [ 1  5  9 ]
    // [ 2  6 10 ]
    // [ 3  7 11 ]
    //
    // dL/dx = 2*x for selected rows, 0 elsewhere.

    checkArray(x.grad(), af::dim4(4, 3), {0, 2, 4, 0, 0, 10, 12, 0, 0, 18, 20, 0},
               "Backward propagation");

    std::cout << "[PASS] Backward propagation\n";
}

void testOverlappingSlices() {
    Tensor x(af::constant(1.0f, 5, 2), true);

    Tensor a = x.slice(1, 4);
    Tensor b = x.slice(2, 5);

    Tensor loss = a.sum() + b.sum();
    loss.backward();

    // Row 0: 0 contributions
    // Row 1: 1 contribution
    // Row 2: 2 contributions
    // Row 3: 2 contributions
    // Row 4: 1 contribution

    checkArray(x.grad(), af::dim4(5, 2), {0, 1, 2, 2, 1, 0, 1, 2, 2, 1}, "Overlapping slices");

    std::cout << "[PASS] Overlapping slice gradients\n";
}

// ==================== Validation Tests ====================

void testInvalidRanges() {
    Tensor x(af::constant(1.0f, 4, 3), true);

    expectThrow<std::invalid_argument>([&]() { x.slice(2, 2); }, "Empty slice");

    expectThrow<std::invalid_argument>([&]() { x.slice(3, 2); }, "Reversed slice");

    expectThrow<std::out_of_range>([&]() { x.slice(-1, 2); }, "Negative start");

    expectThrow<std::out_of_range>([&]() { x.slice(0, 5); }, "End out of bounds");

    expectThrow<std::out_of_range>([&]() { x.slice(4, 5); }, "Start out of bounds");

    std::cout << "[PASS] Invalid range validation\n";
}

void testNoAutograd() {
    Tensor x(af::iota(af::dim4(4, 2), af::dim4(1), f32), false);

    Tensor y = x.slice(1, 3);

    if (y.requires_grad())
        throw std::runtime_error("Slice should not require gradients");

    checkArray(y.data(), af::dim4(2, 2), {1, 2, 5, 6}, "No-autograd forward");

    std::cout << "[PASS] Slicing without autograd\n";
}

// ==================== Mini-batch Tests ====================

void testIncompleteBatch() {
    Tensor x(af::iota(af::dim4(10, 2), af::dim4(1), f32), true);

    constexpr int batchSize = 4;

    int batches = 0;
    int processedSamples = 0;

    for (int start = 0; start < 10; start += batchSize) {
        int end = std::min(start + batchSize, 10);

        Tensor batch = x.slice(start, end);

        if (batch.shape()[0] != end - start)
            throw std::runtime_error("Incorrect batch size");

        if (batch.shape()[1] != 2)
            throw std::runtime_error("Feature dimension changed");

        processedSamples += end - start;
        ++batches;
    }

    if (batches != 3 || processedSamples != 10)
        throw std::runtime_error("Incorrect mini-batch coverage");

    std::cout << "[PASS] Final incomplete mini-batch\n";
}

// ==================== Main ====================

int main() {
    try {
        testForwardSlicing();
        testFullSlice();
        testSingleRow();
        testLastRow();
        testBackwardPropagation();
        testOverlappingSlices();
        testInvalidRanges();
        testNoAutograd();
        testIncompleteBatch();

        std::cout << "\nAll Tensor slicing tests passed!\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[FAIL] " << e.what() << '\n';
        return 1;
    }
}
