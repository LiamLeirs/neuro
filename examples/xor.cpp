
#include "neuro/Tensor.hpp"
#include "neuro/data/DataLoader.hpp"
#include "neuro/data/TensorDataset.hpp"
#include "neuro/nn/Linear.hpp"
#include "neuro/nn/ReLU.hpp"
#include "neuro/nn/Sequential.hpp"
#include "neuro/optim/SGD.hpp"

#include <arrayfire.h>
#include <iostream>

using namespace neuro;

int main() {
    using Matrix = std::initializer_list<std::initializer_list<float>>;
    af::setSeed(42);
    Matrix inputValues = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    Matrix targetValues = {{0}, {1}, {1}, {0}};

    Tensor input(inputValues);
    Tensor target(targetValues);

    TensorDataset dataset({input[0], input[1], input[2], input[3]},
                          {target[0], target[1], target[2], target[3]});

    DataLoader loader(dataset, 4, true);

    nn::Sequential seq(nn::Linear(2, 4), nn::ReLU(), nn::Linear(4, 1));

    SGD sgd(seq.parameters(), 0.01);

    const int NUM_EPOCHS = 5000;

    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        float epochLoss = 0.0f;

        for (auto [x, y] : loader) {
            sgd.zeroGrad();

            Tensor output = seq(x);

            Tensor diff = output - y;
            Tensor loss = (diff * diff).sum() / x.shape()[0];

            epochLoss += loss.data().scalar<float>() * x.shape()[0];

            loss.backward();
            sgd.step();
        }

        epochLoss /= input.shape()[0];

        if (epoch == 0 || (epoch + 1) % 500 == 0) {
            std::cout << "Epoch " << epoch + 1 << "/" << NUM_EPOCHS << " | Loss: " << epochLoss
                      << '\n';
        }
    }

    // Evaluate the trained model on the full dataset.
    Tensor predictions = seq(input);

    std::cout << "\nFinal XOR predictions:\n";

    af::array result = predictions.data();
    af::array inputs = input.data();

    for (int i = 0; i < input.shape()[0]; i++) {
        float a = inputs(i, 0).scalar<float>();
        float b = inputs(i, 1).scalar<float>();
        float prediction = result(i, 0).scalar<float>();

        std::cout << a << " XOR " << b << " -> " << prediction << '\n';
    }

    return 0;
}
