
#include "neuro/Tensor.hpp"
#include "neuro/data/DataLoader.hpp"
#include "neuro/data/MNISTDataset.hpp"
#include "neuro/data/TensorDataset.hpp"
#include "neuro/nn/Linear.hpp"
#include "neuro/nn/ReLU.hpp"
#include "neuro/nn/Sequential.hpp"
#include "neuro/nn/loss/CrossEntropyLoss.hpp"
#include "neuro/optim/SGD.hpp"

#include <arrayfire.h>
#include <cstddef>
#include <iostream>
#include <ostream>

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

    for (auto [x, y] : train_loader) {
        Tensor output = model(x);
        Tensor loss = nn::CrossEntropyLoss()(output, y);
        std::cout << output.shape() << std::endl;
    }

    return 0;
}
