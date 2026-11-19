
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
        float totalLoss = 0.0f;
        for (int i = 0; i < input.data().elements(); i++) {
            Tensor predicted = w * input[i] + b;
            // MSE Loss
            Tensor error = predicted - target[i];
            Tensor loss = error * error;
            totalLoss += loss.data().scalar<float>();
            optim.zeroGrad();
            loss.backward();
            optim.step();
        }
        if (epoch % 20 == 0) {
            std::cout << "Epoch " << epoch << " | Loss: " << totalLoss / input.data().elements()
                      << " | w: " << w.data().scalar<float>()
                      << " | b: " << b.data().scalar<float>() << '\n';
        }
    }

    return 0;
}
