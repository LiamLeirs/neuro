
#include "neuro/GradMode.hpp"
#include "neuro/Tensor.hpp"
#include "neuro/data/DataLoader.hpp"
#include "neuro/data/MNISTDataset.hpp"
#include "neuro/metrics/Accuracy.hpp"
#include "neuro/metrics/Mean.hpp"
#include "neuro/nn/Linear.hpp"
#include "neuro/nn/ReLU.hpp"
#include "neuro/nn/Sequential.hpp"
#include "neuro/nn/loss/CrossEntropyLoss.hpp"
#include "neuro/optim/Adam.hpp"

#include <cstddef>
#include <iostream>

using namespace neuro;

int main() {
    std::string directory = "data/mnist";
    MNISTDataset train_dataset = MNISTDataset(directory + "/train-images.idx3-ubyte",
                                              directory + "/train-labels.idx1-ubyte");
    MNISTDataset test_dataset =
        MNISTDataset(directory + "/t10k-images.idx3-ubyte", directory + "/t10k-labels.idx1-ubyte");

    const std::size_t BATCH_SIZE = 128;
    DataLoader train_loader = DataLoader(train_dataset, BATCH_SIZE, true);
    DataLoader test_loader = DataLoader(test_dataset, BATCH_SIZE, false);

    nn::Sequential model(nn::Linear(784, 128), nn::ReLU(), nn::Linear(128, 10));
    const float LR = 1e-3;
    const int NUM_EPOCHS = 10;
    nn::CrossEntropyLoss criterion;
    Adam optim(model.parameters(), LR);
    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
        metrics::Mean mean;
        for (auto [x, y] : train_loader) {
            optim.zeroGrad();

            Tensor output = model(x);
            Tensor loss = criterion(output, y);

            loss.backward();
            optim.step();

            const std::size_t batch_size = x.shape()[0];

            mean.update(loss, batch_size);
        }

        std::cout << "Epoch [" << epoch + 1 << "/" << NUM_EPOCHS << "] - Loss: " << mean.compute()
                  << '\n';
    }

    metrics::Accuracy accuracy;

    {
        NoGradGuard guard;
        for (auto [x, y] : test_loader) {
            Tensor predictions = model(x);

            accuracy.update(predictions, y);
        }
    }

    std::cout << "\nFinal Test Accuracy: " << accuracy.compute() * 100 << "%\n";

    return 0;
}
