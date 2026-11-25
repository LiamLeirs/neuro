
#include "neuro/Trainer.hpp"
#include "neuro/data/CIFAR10Dataset.hpp"
#include "neuro/data/DataLoader.hpp"
#include "neuro/metrics/Accuracy.hpp"
#include "neuro/nn/Conv2D.hpp"
#include "neuro/nn/Flatten.hpp"
#include "neuro/nn/Linear.hpp"
#include "neuro/nn/MaxPool2D.hpp"
#include "neuro/nn/ReLU.hpp"
#include "neuro/nn/Sequential.hpp"
#include "neuro/nn/loss/CrossEntropyLoss.hpp"
#include "neuro/optim/Adam.hpp"

#include <cstddef>
#include <iostream>
#include <string>

using namespace neuro;

int main() {
    std::string directory = "data/cifar10/";
    std::vector<std::string> train_files;
    for (int i = 1; i <= 5; ++i) {
        train_files.push_back(directory + "data_batch_" + std::to_string(i) + ".bin");
    }
    CIFAR10Dataset train_dataset = CIFAR10Dataset(train_files);
    CIFAR10Dataset test_dataset = CIFAR10Dataset({directory + "test_batch.bin"});

    const std::size_t BATCH_SIZE = 128;
    DataLoader train_loader = DataLoader(train_dataset, BATCH_SIZE, true);
    DataLoader test_loader = DataLoader(test_dataset, BATCH_SIZE, false);

    nn::Sequential model(nn::Conv2D(3, 32, 3, 1, 1), nn::ReLU(), nn::MaxPool2D(2),
                         nn::Conv2D(32, 64, 3, 1, 1), nn::ReLU(), nn::MaxPool2D(2),
                         nn::Conv2D(64, 128, 3, 1, 1), nn::ReLU(), nn::MaxPool2D(2), nn::Flatten(),
                         nn::Linear(2048, 128), nn::ReLU(), nn::Linear(128, 10));

    const float LR = 1e-3;
    const int NUM_EPOCHS = 20;
    nn::CrossEntropyLoss criterion;
    Adam optim(model.parameters(), LR);

    Trainer trainer(model, optim, criterion);

    trainer.addMetric<metrics::Accuracy>("accuracy");

    auto history = trainer.fit(train_loader, NUM_EPOCHS);

    auto results = trainer.evaluate(test_loader);

    std::cout << "\nFinal Test Loss: " << results.at("loss") << '\n';
    std::cout << "Final Test Accuracy: " << results.at("accuracy") * 100.0 << "%\n";

    return 0;
}
