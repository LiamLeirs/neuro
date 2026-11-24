#include "neuro/Trainer.hpp"

#include "neuro/GradMode.hpp"
#include "neuro/metrics/Mean.hpp"

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace neuro {
Trainer::Trainer(nn::Module& model, Optimizer& optim, LossFunction criterion)
    : model_(model), optim_(optim), criterion_(std::move(criterion)) {
    if (!criterion_) {
        throw std::invalid_argument("Provided criterion is invalid");
    }
}

void Trainer::fit(DataLoader& loader, std::size_t num_epochs) {
    for (std::size_t epoch = 0; epoch < num_epochs; epoch++) {
        double epoch_loss = trainEpoch(loader);
        std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs << "] - Loss: " << epoch_loss
                  << '\n';
    }
}

double Trainer::trainEpoch(DataLoader& loader) {
    metrics::Mean mean;
    for (auto [x, y] : loader) {
        optim_.zeroGrad();

        Tensor output = model_(x);
        Tensor loss = criterion_(output, y);

        loss.backward();
        optim_.step();

        const std::size_t batch_size = x.shape()[0];

        mean.update(loss, batch_size);
    }
    return mean.compute();
}

double Trainer::evaluate(DataLoader& loader) {
    metrics::Mean mean;
    NoGradGuard guard;
    for (auto [x, y] : loader) {
        const std::size_t batch_size = x.shape()[0];
        Tensor output = model_(x);
        Tensor loss = criterion_(output, y);
        mean.update(loss, batch_size);
    }

    return mean.compute();
};
} // namespace neuro