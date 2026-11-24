
#include "neuro/GradMode.hpp"
#include "neuro/Tensor.hpp"
#include "neuro/Trainer.hpp"
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

    Trainer trainer(model, optim, criterion);
    trainer.fit(train_loader, NUM_EPOCHS);

    double loss = trainer.evaluate(test_loader);

    std::cout << "\nFinal Test Loss: " << loss << "\n";

    return 0;
}
