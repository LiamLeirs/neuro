
#include "../include/neuro/Tensor.hpp"

#include <arrayfire.h>
#include <cmath>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using neuro::Tensor;

using Matrix = std::initializer_list<std::initializer_list<float>>;

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

void testBroadcastColumns() {
    // A: (3,4), B: (3,1)
    Tensor a(Matrix{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}, true);

    Tensor b(Matrix{{10}, {20}, {30}}, true);

    Tensor y = a + b;

    checkArray(y.data(), af::dim4(3, 4), {11, 25, 39, 12, 26, 40, 13, 27, 41, 14, 28, 42},
               "Column broadcasting forward");

    y.sum().backward();

    checkArray(a.grad(), af::dim4(3, 4), std::vector<float>(12, 1.0f),
               "Column broadcasting a.grad");

    checkArray(b.grad(), af::dim4(3, 1), {4, 4, 4}, "Column broadcasting b.grad");

    std::cout << "[PASS] Broadcasting across columns\n";
}

void testBroadcastRows() {
    // A: (3,4), B: (1,4)
    Tensor a(Matrix{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}, true);

    Tensor b(Matrix{{10, 20, 30, 40}}, true);

    Tensor y = a + b;

    checkArray(y.data(), af::dim4(3, 4), {11, 15, 19, 22, 26, 30, 33, 37, 41, 44, 48, 52},
               "Row broadcasting forward");

    y.sum().backward();

    checkArray(a.grad(), af::dim4(3, 4), std::vector<float>(12, 1.0f), "Row broadcasting a.grad");

    checkArray(b.grad(), af::dim4(1, 4), {3, 3, 3, 3}, "Row broadcasting b.grad");

    std::cout << "[PASS] Broadcasting across rows\n";
}

void testBroadcastSubtraction() {
    // A: (2,3), B: (2,1)
    Tensor a(Matrix{{5, 6, 7}, {8, 9, 10}}, true);

    Tensor b(Matrix{{1}, {2}}, true);

    Tensor y = b - a;

    checkArray(y.data(), af::dim4(2, 3), {-4, -6, -5, -7, -6, -8}, "Subtraction forward");

    y.sum().backward();

    checkArray(a.grad(), af::dim4(2, 3), std::vector<float>(6, -1.0f), "Subtraction a.grad");

    checkArray(b.grad(), af::dim4(2, 1), {3, 3}, "Subtraction b.grad");
    std::cout << "[PASS] Multidimensional subtraction\n";
}

void testBroadcastMultiplication() {
    // A: (3,2), B: (3,1)
    Tensor a(Matrix{{1, 2}, {3, 4}, {5, 6}}, true);

    Tensor b(Matrix{{2}, {3}, {4}}, true);

    Tensor y = a * b;

    checkArray(y.data(), af::dim4(3, 2), {2, 9, 20, 4, 12, 24}, "Multiplication forward");

    y.sum().backward();

    checkArray(a.grad(), af::dim4(3, 2), {2, 3, 4, 2, 3, 4}, "Multiplication a.grad");

    checkArray(b.grad(), af::dim4(3, 1), {3, 7, 11}, "Multiplication b.grad");

    std::cout << "[PASS] Multidimensional multiplication\n";
}

void testBroadcastDivision() {
    // A: (2,3), B: (2,1)
    Tensor a(Matrix{{2, 4, 6}, {8, 12, 16}}, true);

    Tensor b(Matrix{{2}, {4}}, true);

    Tensor y = a / b;

    checkArray(y.data(), af::dim4(2, 3), {1, 2, 2, 3, 3, 4}, "Division forward");

    y.sum().backward();

    checkArray(a.grad(), af::dim4(2, 3), {0.5f, 0.25f, 0.5f, 0.25f, 0.5f, 0.25f},
               "Division a.grad");

    checkArray(b.grad(), af::dim4(2, 1), {-3, -2.25f}, "Division b.grad");

    std::cout << "[PASS] Multidimensional division\n";
}

void testMultipleBroadcastAxes() {
    // A: (2,3,4), B: (1,3,1)
    af::array aData = af::constant(1.0f, 2, 3, 4);
    af::array bData = af::constant(2.0f, 1, 3, 1);

    Tensor a(aData, true);
    Tensor b(bData, true);

    Tensor y = a * b;

    checkArray(y.data(), af::dim4(2, 3, 4), std::vector<float>(24, 2.0f), "Multiple axes forward");

    y.sum().backward();

    checkArray(a.grad(), af::dim4(2, 3, 4), std::vector<float>(24, 2.0f), "Multiple axes a.grad");

    checkArray(b.grad(), af::dim4(1, 3, 1), {8, 8, 8}, "Multiple axes b.grad");

    std::cout << "[PASS] Multiple broadcast axes\n";
}

void testIncompatibleShapes() {
    Tensor a(af::constant(1.0f, 3, 4), true);
    Tensor b(af::constant(2.0f, 2, 4), true);

    bool threw = false;

    try {
        Tensor y = a + b;
        y.data().eval();
    } catch (const af::exception&) {
        threw = true;
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    if (!threw)
        throw std::runtime_error("Incompatible shapes should throw an exception");

    std::cout << "[PASS] Incompatible shapes\n";
}

int main() {
    try {
        testBroadcastColumns();
        testBroadcastRows();
        testBroadcastSubtraction();
        testBroadcastMultiplication();
        testBroadcastDivision();
        testMultipleBroadcastAxes();
        testIncompatibleShapes();

        std::cout << "\nAll general broadcasting tests passed!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] " << e.what() << '\n';
        return 1;
    }
}
