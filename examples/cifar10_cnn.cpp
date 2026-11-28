
#include "neuro/Trainer.hpp"
#include "neuro/data/CIFAR10Dataset.hpp"
#include "neuro/data/DataLoader.hpp"
#include "neuro/data/TransformDataset.hpp"
#include "neuro/data/transforms/Compose.hpp"
#include "neuro/data/transforms/Normalize.hpp"
#include "neuro/data/transforms/RandomCrop.hpp"
#include "neuro/data/transforms/RandomHorizontalFlip.hpp"
#include "neuro/metrics/Accuracy.hpp"
#include "neuro/nn/BatchNorm1D.hpp"
#include "neuro/nn/BatchNorm2D.hpp"
#include "neuro/nn/Conv2D.hpp"
#include "neuro/nn/Dropout.hpp"
#include "neuro/nn/Flatten.hpp"
#include "neuro/nn/Linear.hpp"
#include "neuro/nn/MaxPool2D.hpp"
#include "neuro/nn/ReLU.hpp"
#include "neuro/nn/Sequential.hpp"
#include "neuro/nn/loss/CrossEntropyLoss.hpp"
#include "neuro/optim/AdamW.hpp"
#include "neuro/optim/CosineAnnealingLR.hpp"
#include "neuro/viz/Plot.hpp"

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

using namespace neuro;
using namespace transforms;

int main() {
    std::string directory = "data/cifar10/";
    std::vector<std::string> train_files;
    for (int i = 1; i <= 4; ++i) {
        train_files.push_back(directory + "data_batch_" + std::to_string(i) + ".bin");
    }
    CIFAR10Dataset train_dataset = CIFAR10Dataset(train_files);
    CIFAR10Dataset val_dataset = CIFAR10Dataset({directory + "data_batch_5.bin"});
    CIFAR10Dataset test_dataset = CIFAR10Dataset({directory + "test_batch.bin"});

    const std::vector<float> mean{0.4914f, 0.4822f, 0.4465f};
    const std::vector<float> stddev{0.2470f, 0.2435f, 0.2616f};

    TransformDataset train(train_dataset, Compose(RandomCrop(32, 32, 4), RandomHorizontalFlip(0.5),
                                                  Normalize(mean, stddev)));

    TransformDataset val(val_dataset, Normalize(mean, stddev));

    TransformDataset test(test_dataset, Normalize(mean, stddev));

    const std::size_t BATCH_SIZE = 128;
    DataLoader train_loader(train, BATCH_SIZE, true);
    DataLoader val_loader(val, BATCH_SIZE, false);
    DataLoader test_loader(test, BATCH_SIZE, false);

    nn::Sequential model(nn::Conv2D(3, 32, 3, 1, 1), nn::BatchNorm2D(32), nn::ReLU(),
                         nn::MaxPool2D(2), nn::Conv2D(32, 64, 3, 1, 1), nn::BatchNorm2D(64),
                         nn::ReLU(), nn::MaxPool2D(2), nn::Conv2D(64, 128, 3, 1, 1),
                         nn::BatchNorm2D(128), nn::ReLU(), nn::MaxPool2D(2), nn::Flatten(),
                         nn::Linear(2048, 128), nn::BatchNorm1D(128), nn::ReLU(), nn::Dropout(0.5),
                         nn::Linear(128, 10));

    const double LR = 1e-3;
    const double WEIGHT_DECAY = 1e-4;
    const std::size_t NUM_EPOCHS = 50;

    nn::CrossEntropyLoss criterion;

    AdamW optim(model.parameters(), LR, 0.9, 0.999, 1e-8, WEIGHT_DECAY);
    CosineAnnealingLR scheduler(optim, NUM_EPOCHS, 1e-5);

    Trainer trainer(model, optim, criterion);

    trainer.addMetric<metrics::Accuracy>("accuracy");
    trainer.setScheduler(scheduler);

    auto history = trainer.fit(train_loader, val_loader, NUM_EPOCHS);

    auto results = trainer.evaluate(test_loader);

    neuro::viz::Plot plot;

    plot.addSeries(history, "loss", "Training")
        .addSeries(history, "val_loss", "Validation")
        .title("CIFAR-10 Loss")
        .xlabel("Epoch")
        .ylabel("Loss")
        .save("cifar10_loss.svg");

    plot.clear()
        .addSeries(history, "accuracy", "Training")
        .addSeries(history, "val_accuracy", "Validation")
        .title("CIFAR-10 Accuracy")
        .xlabel("Epoch")
        .ylabel("Accuracy")
        .save("cifar10_accuracy.svg");

    plot.clear()
        .addSeries(history, "lr", "Learning rate")
        .title("CIFAR-10 Learning Rate")
        .xlabel("Epoch")
        .ylabel("Learning Rate")
        .save("cifar10_lr.svg");

    std::cout << "\nTraining plots saved:\n"
              << "  cifar10_loss.svg\n"
              << "  cifar10_accuracy.svg\n"
              << "  cifar10_lr.svg\n";

    std::cout << "\nTest loss: " << results.at("loss") << '\n';
    std::cout << "Test accuracy: " << results.at("accuracy") * 100.0 << "%\n";

    return 0;
}
