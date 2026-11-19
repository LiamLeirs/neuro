
#include "neuro/Tensor.hpp"
#include "neuro/optim/SGD.hpp"

#include <algorithm>
#include <initializer_list>
#include <iostream>

using namespace neuro;

int main() {
    using Matrix = std::initializer_list<std::initializer_list<float>>;

    // Two weights and one bias
    Tensor w(Matrix{{0.0f}, {0.0f}}, true);

    Tensor b({0.0f}, true);

    // 8 samples, 2 features
    Tensor input(Matrix{{-2, -1}, {-1, 2}, {0, -2}, {1, 1}, {2, 0}, {3, -1}, {-3, 2}, {2, 3}});

    // y = 2*x1 - 3*x2 + 1
    Tensor target(Matrix{{0}, {-7}, {7}, {0}, {5}, {10}, {-11}, {-4}});

    SGD optim({w, b}, 1e-2);

    int BATCH_SIZE = 3;

    for (int epoch = 0; epoch < 500; epoch++) {
        float totalLoss = 0;
        for (int start = 0; start < input.data().dims(0); start += BATCH_SIZE) {
            int end = std::min(start + BATCH_SIZE, (int)input.data().dims(0));
            Tensor predicted = input.slice(start, end).matmul(w) + b;

            Tensor error = predicted - target.slice(start, end);
            Tensor loss = (error * error).sum() / (end - start);
            totalLoss += loss.data().scalar<float>() * (end - start);

            optim.zeroGrad();
            loss.backward();
            optim.step();
        }
        if (epoch % 50 == 0) {
            std::cout << "Epoch " << epoch << " | Loss: " << totalLoss / input.data().dims(0)
                      << " | w1: " << w.data()(0).scalar<float>()
                      << " | w2: " << w.data()(1).scalar<float>()
                      << " | b: " << b.data().scalar<float>() << '\n';
        }
    }

    return 0;
}
