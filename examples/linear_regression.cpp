
#include "neuro/Tensor.hpp"
#include "neuro/optim/SGD.hpp"

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

    for (int epoch = 0; epoch < 500; epoch++) {
        Tensor predicted = input.matmul(w) + b;

        Tensor error = predicted - target;
        Tensor loss = (error * error).sum() / target.data().elements();

        optim.zeroGrad();
        loss.backward();
        optim.step();

        if (epoch % 50 == 0) {
            std::cout << "Epoch " << epoch << " | Loss: " << loss.data().scalar<float>()
                      << " | w1: " << w.data()(0).scalar<float>()
                      << " | w2: " << w.data()(1).scalar<float>()
                      << " | b: " << b.data().scalar<float>() << '\n';
        }
    }

    return 0;
}
