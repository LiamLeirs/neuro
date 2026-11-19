
#include "neuro/Tensor.hpp"
#include "neuro/optim/SGD.hpp"

#include <iostream>

using namespace neuro;

int main() {
    Tensor w({0.0f}, true);
    Tensor b({0.0f}, true);

    Tensor input = {-2, -1, 0, 1, 2};
    Tensor target = {-3, -1, 1, 3, 5};

    SGD optim = SGD({w, b}, 1e-2);
    for (int epoch = 0; epoch < 200; epoch++) {
        Tensor predicted = input.matmul(w) + b;
        // MSE Loss
        Tensor error = predicted - target;
        Tensor loss = error * error;
        loss = loss.sum() / input.data().elements();
        optim.zeroGrad();
        loss.backward();
        optim.step();
        if (epoch % 20 == 0) {
            std::cout << "Epoch " << epoch << " | Loss: " << loss.data().scalar<float>()
                      << " | w: " << w.data().scalar<float>()
                      << " | b: " << b.data().scalar<float>() << '\n';
        }
    }

    return 0;
}
