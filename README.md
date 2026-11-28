# Neuro

![CI](https://github.com/LiamLeirs/neuro/actions/workflows/ci.yml/badge.svg)

![C++20](https://img.shields.io/badge/C%2B%2B-20-blue)

![CUDA](https://img.shields.io/badge/CUDA-enabled-76B900)

![CPU](https://img.shields.io/badge/CPU-enabled-blue)

![License](https://img.shields.io/badge/license-MIT-blue)

![Platform](https://img.shields.io/badge/platform-Linux-lightgrey)

Neuro is a GPU-first deep learning library written in C++20, built on top of ArrayFire with CUDA and CPU backend support. It implements a custom reverse-mode automatic differentiation engine and provides a PyTorch-inspired API for building, training and evaluating neural networks.

The project aims to explore the internal workings of deep learning frameworks, from tensor operations and computational graphs to gradient propagation and model optimization.

Neuro is currently in early development and targets Linux systems with either a CPU or a CUDA-compatible NVIDIA GPU.

## Features

- **GPU-accelerated tensors** — Tensor arithmetic, broadcasting, matrix multiplication, reductions, indexing, reshaping and dimension reordering powered by ArrayFire.
- **Configurable backend** — CUDA acceleration by default, with CPU support selected at build time.
- **Automatic differentiation** — Dynamic computational graphs, reverse-mode gradient propagation and gradient tracking control.
- **Neural networks** — Fully connected and convolutional layers, pooling, flattening, activation functions, dropout, batch normalization and composable sequential models.
- **Optimizers** — Stochastic Gradient Descent (SGD), Adam and AdamW, with weight decay support.
- **Loss functions** — Mean Squared Error (MSE), Mean Absolute Error (MAE) and Cross-Entropy Loss.
- **Data utilities** — Dataset abstractions, batched data loading, shuffling and optimized MNIST and CIFAR-10 data loading, composable transforms, normalization and random image augmentation.
- **Training utilities** — Reusable training and evaluation loops, per-epoch validation, extensible metrics, learning rate scheduling, early stopping and training history.
- **Visualization** — Custom SVG plotting with automatic ticks, grid lines, legends, training history integration and Linux display support.
- **CMake integration** — Installable CMake package with support for `find_package(Neuro)`.
- **Unit testing** — GoogleTest coverage of tensor operations, automatic differentiation, neural network components, datasets, training utilities and visualization.

## Quick Start

### Automatic Differentiation

Neuro automatically constructs a computational graph during the forward pass and computes gradients through reverse-mode automatic differentiation.

```cpp
#include <neuro/Tensor.hpp>
#include <iostream>
using namespace neuro;
int main() {
    Tensor x({2.0f}, true);
    Tensor y({3.0f}, true);
    Tensor z = x * y + x;
    z.backward();
    std::cout << "Result: " << z << '\n';
    std::cout << "Gradient of x: " << x.grad().scalar<float>() << '\n';
    std::cout << "Gradient of y: " << y.grad().scalar<float>() << '\n';
}
```

The resulting value is `8`, with gradients of `4` and `2` with respect to `x` and `y`, respectively.

### Training a Neural Network

Neuro provides a high-level training API for building and training neural networks with minimal boilerplate.

```cpp
#include <neuro/Tensor.hpp>
#include <neuro/Trainer.hpp>
#include <neuro/data/TensorDataset.hpp>
#include <neuro/data/DataLoader.hpp>
#include <neuro/nn/Linear.hpp>
#include <neuro/nn/Sequential.hpp>
#include <neuro/nn/Tanh.hpp>
#include <neuro/nn/Sigmoid.hpp>
#include <neuro/nn/loss/MSELoss.hpp>
#include <neuro/optim/Adam.hpp>
using namespace neuro;
int main() {
    TensorDataset dataset(
        {
            Tensor{0.f, 0.f},
            Tensor{0.f, 1.f},
            Tensor{1.f, 0.f},
            Tensor{1.f, 1.f}
        },
        {
            Tensor{0.f},
            Tensor{1.f},
            Tensor{1.f},
            Tensor{0.f}
        }
    );
    DataLoader loader(dataset, 4, true);
    nn::Sequential model(
        nn::Linear(2, 8),
        nn::Tanh(),
        nn::Linear(8, 1),
        nn::Sigmoid()
    );
    nn::MSELoss criterion;
    Adam optimizer(model.parameters(), 0.01);
    Trainer trainer(model, optimizer, criterion);
    trainer.fit(loader, 1000);
}
```

The example trains a small multilayer perceptron to learn the XOR function. Gradient computation, parameter updates and loss tracking are handled automatically by the Trainer.

### Validation, Scheduling and Early Stopping

Trainer supports a separate validation loader and optional learning rate scheduling and early stopping.

```cpp
// After constructing a model, optimizer, trainer and data loaders:
CosineAnnealingLR scheduler(optimizer, 50, 1e-5);
EarlyStopping stopping(PlateauMode::Min, 10);

trainer.setScheduler(scheduler);
trainer.setEarlyStopping(stopping, "val_loss");

auto history = trainer.fit(train_loader, val_loader, 50);
```

Training history includes training metrics, validation metrics prefixed with `val_`, and the learning rate used during each epoch (`lr`). StepLR, ExponentialLR, CosineAnnealingLR and ReduceLROnPlateau are supported. Metric-based scheduling and early stopping can monitor loss or registered training and validation metrics.

Trainer borrows the scheduler and EarlyStopping instances; they must remain alive while registered and in use. EarlyStopping state persists across calls to `fit()`. Call `reset()` on the EarlyStopping instance to resume after stopping, or detach it using `clearEarlyStopping()`. The stopping epoch is included in the history. Best weights are not automatically restored.

### Dataset Transforms

Transforms can be composed and applied lazily to dataset samples or batches. For example, a CIFAR-10 training pipeline can use:

```cpp
#include <neuro/data/TransformDataset.hpp>
#include <neuro/data/transforms/Compose.hpp>
#include <neuro/data/transforms/RandomCrop.hpp>
#include <neuro/data/transforms/RandomHorizontalFlip.hpp>
#include <neuro/data/transforms/Normalize.hpp>

using namespace neuro;
using namespace neuro::transforms;

// Given an existing CIFAR-10 training dataset:
TransformDataset train(
    train_dataset,
    Compose(
        RandomCrop(32, 32, 4),
        RandomHorizontalFlip(0.5),
        Normalize({0.4914f, 0.4822f, 0.4465f},
                  {0.2470f, 0.2435f, 0.2616f})
    )
);
```

Validation and test datasets should use normalization without random augmentation. TransformDataset borrows the underlying dataset and owns its transforms. Batch-capable pipelines use optimized batch processing; other pipelines fall back to individual samples.

### Visualizing Training History

The plotting API accepts training history directly or arbitrary numerical series. It renders SVG files without an additional plotting dependency.

```cpp
#include <neuro/viz/Plot.hpp>

// After constructing a trainer and data loader:
auto history = trainer.fit(loader, 10);

neuro::viz::Plot plot;
plot.addSeries(history, "loss")
    .title("Training Loss")
    .xlabel("Epoch")
    .ylabel("Loss");

plot.save("loss.svg");
plot.show();
```

Multiple series can be added to the same plot. Use `clear()` to reset the series and labels before creating another plot. `save()` and `show()` do not modify the plot.

`show()` requires Linux, `xdg-open` and a graphical environment with an SVG-capable default application. It may block depending on the viewer. Successful calls retain a temporary SVG in `/tmp` so the viewer can access it. SVG export does not require a graphical environment.

## Requirements

Neuro currently targets Linux and supports ArrayFire's CUDA and CPU backends.

### System Requirements

- Linux
- For CUDA builds only: a CUDA-compatible NVIDIA GPU and compatible NVIDIA drivers

### Build Dependencies

- C++ compiler with C++20 support
- CMake 3.24 or newer
- ArrayFire with support for the selected backend (CPU or CUDA)
- NVIDIA CUDA Toolkit (CUDA builds only)

### Optional Dependencies

- Ninja — recommended build system
- GoogleTest — automatically downloaded by CMake when unit testing is enabled

### Development Environment

Neuro has been developed and tested using the following environment:

- CachyOS (Arch Linux)
- NVIDIA GeForce RTX 3070 (8 GB)
- CUDA 13.5
- ArrayFire 3.10.1

Docker builds have also been verified locally on Ubuntu 24.04 with ArrayFire 3.10.0 and CUDA Toolkit 12.8.1, including CPU test execution. Other environments may work but have not been validated.

## Building from Source

Ensure that all required dependencies are installed before building Neuro.

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/LiamLeirs/neuro
cd neuro
```

Configure and build the library in Release mode:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

By default, only the Neuro library is built. Examples and unit tests can be enabled through the following CMake options:

| Option | Default | Description |
|--------|---------|-------------|
| `NEURO_BACKEND` | `CUDA` | Select `CUDA` or `CPU` |
| `NEURO_BUILD_EXAMPLES` | `OFF` | Build example applications |
| `NEURO_BUILD_TESTS` | `OFF` | Build unit tests |

To build Neuro with both enabled:

```bash
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DNEURO_BUILD_EXAMPLES=ON \
    -DNEURO_BUILD_TESTS=ON
cmake --build build -j
```

### Backend Selection

CUDA is the default backend. To build for CPU instead, use:

```bash
cmake -S . -B build/cpu -DNEURO_BACKEND=CPU -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpu -j
```

The public API is identical for both backends. The backend is selected at configuration time, not at runtime. Installed packages retain the backend selected when building Neuro.

### Locating ArrayFire

If CMake cannot locate ArrayFire automatically, specify its installation prefix:

```bash
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=/path/to/arrayfire
```

## Installation

After building Neuro, install the library, public headers and CMake package configuration:

```bash
sudo cmake --install build
```

Alternatively, install Neuro to a custom location without requiring administrator privileges:

```bash
cmake --install build --prefix "$HOME/.local"
```

Once installed, Neuro can be consumed by external CMake projects without manually configuring include directories or linking dependencies.

## Using Neuro in Your Project

Once Neuro is installed, it can be integrated into any CMake-based C++ project using `find_package()`.

### CMake Integration

```cmake
cmake_minimum_required(VERSION 3.24)
project(my_project LANGUAGES CXX)
find_package(Neuro 0.3 CONFIG REQUIRED)
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE Neuro::Neuro)
```

The `Neuro::Neuro` target automatically propagates the required include directories, C++20 compiler requirements and ArrayFire dependency.

### Example Application

```cpp
#include <neuro/Tensor.hpp>
#include <iostream>
int main() {
    neuro::Tensor x{1.0f, 2.0f, 3.0f};
    neuro::Tensor y = x * 2.0f;
    std::cout << y << '\n';
    return 0;
}
```

### Building Your Application

If Neuro is installed in a standard system location:

```bash
cmake -S . -B build
cmake --build build
./build/my_app
```

If Neuro is installed in a custom location, specify its installation prefix:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/neuro
cmake --build build
./build/my_app
```

ArrayFire is resolved automatically during CMake configuration, provided that it is installed and discoverable.

### Version Compatibility

Neuro uses semantic versioning. During pre-1.0 development, the CMake package declares compatibility within the same minor version.

For example, a project requesting Neuro 0.1.0 can accept version 0.1.1, but not version 0.2.0.

## Examples

Neuro includes several example applications demonstrating its tensor, automatic differentiation and neural network APIs.

| Example | Description |
|---------|-------------|
| Linear Regression | Learn a linear relationship using gradient-based optimization. |
| XOR Classification | Train a multilayer perceptron to learn a non-linear function. |
| MNIST Classification | Train and evaluate a multilayer perceptron on handwritten digits. |
| MNIST CNN | Train a convolutional network on handwritten digits. |
| CIFAR-10 CNN | Train a convolutional network on color images and export training curves. |

### Running the Examples

Build Neuro with examples enabled:

```bash
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DNEURO_BUILD_EXAMPLES=ON
cmake --build build -j
```

Run the examples from the repository root:

```bash
./build/examples/neuro_linear_regression
./build/examples/neuro_xor
./build/examples/neuro_mnist
./build/examples/neuro_cnn
./build/examples/neuro_cifar10
```

### MNIST Classification

The MNIST example demonstrates an end-to-end training pipeline using Neuro's dataset, data loading, neural network, optimization and training APIs.

#### Model Architecture

| Layer | Output Features |
|-------|-----------------|
| Input | 784 |
| Linear + ReLU | 128 |
| Linear | 10 |

#### Training Configuration

| Parameter | Value |
|-----------|-------|
| Training samples | 60,000 |
| Test samples | 10,000 |
| Trainable parameters | 101,770 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 128 |
| Epochs | 10 |
| Loss function | Cross-Entropy Loss |

#### Results

| Metric | Value |
|--------|-------|
| Final training loss | 0.0324 |
| Final training accuracy (during training) | 99.11% |
| Test loss | 0.0815 |
| Test accuracy | **97.57%** |

The model correctly classified 9,757 out of 10,000 test images after 10 training epochs.

#### Dataset Setup

The MNIST dataset is not included in the repository. Download and extract the original IDX dataset files into the following directory:

```text
data/mnist/
├── train-images.idx3-ubyte
├── train-labels.idx1-ubyte
├── t10k-images.idx3-ubyte
└── t10k-labels.idx1-ubyte
```

Run the MNIST example from the repository root to ensure the dataset files can be located.

### Convolutional Neural Networks

Neuro supports differentiable convolution and pooling operations, NCHW image batches and convolutional modules. The MNIST dataset supports both flattened and image layouts.

The CIFAR-10 example uses three convolutional blocks with batch normalization, ReLU and max pooling, followed by flattening, a fully connected hidden layer with batch normalization and dropout, and a classification layer. It trains using AdamW, cosine learning rate annealing and Cross-Entropy Loss, with random cropping, horizontal flipping and normalization. Training and validation loss, accuracy and learning rate are exported as SVG files.

#### Example Results

| Dataset | Model | Test accuracy | Test loss |
|---------|-------|--------------:|----------:|
| MNIST | MLP | 97.57% | 0.0815 |
| MNIST | CNN | 98.82% | 0.0351 |
| CIFAR-10 | CNN (v0.2.0) | 73.24% | 1.3498 |
| CIFAR-10 | CNN (v0.3.0) | **84.09%** | 0.4668 |

These are results from individual local training runs, not formal performance benchmarks. Results may vary between runs.

#### CIFAR-10 Training Configuration

| Parameter | Value |
|-----------|-------|
| Training samples | 40,000 |
| Validation samples | 10,000 |
| Test samples | 10,000 |
| Trainable parameters | 357,514 |
| Optimizer | AdamW |
| Weight decay | 0.0001 |
| Initial learning rate | 0.001 |
| Scheduler | CosineAnnealingLR |
| Minimum learning rate | 0.00001 |
| Dropout | 0.5 |
| Augmentation | Random crop (padding 4), horizontal flip (p = 0.5) |
| Batch size | 128 |
| Epochs | 50 |
| Loss function | Cross-Entropy Loss |

The first four CIFAR-10 training batches are used for training, the fifth for validation, and the separate test batch for final evaluation. Normalization uses reference statistics from the full original training set. The test set is not used for training or validation.

The v0.3.0 run took 27 minutes and 16 seconds on an RTX 3070 and reached 84.09% test accuracy, an improvement of 10.85 percentage points over v0.2.0. The comparison is not a controlled ablation: the training split, architecture, augmentation, optimizer and epoch count differ.

#### CIFAR-10 Dataset Setup

Download the binary version of CIFAR-10 and extract it from the repository root:

```bash
mkdir -p data/cifar10
wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz -O /tmp/cifar-10-binary.tar.gz
tar -xzf /tmp/cifar-10-binary.tar.gz -C data/cifar10 --strip-components=1
```

The directory should contain `data_batch_1.bin` through `data_batch_5.bin` and `test_batch.bin`.

## Testing

Neuro includes a comprehensive unit test suite built with GoogleTest. The tests cover:

- Tensor operations and broadcasting
- Automatic differentiation and gradient accumulation
- Gradient tracking control
- Neural network layers and activation functions
- Loss functions and optimizers
- Datasets and data loading
- Metrics, validation, learning rate schedulers and early stopping
- Dataset transforms and image augmentation
- Dropout and batch normalization
- Convolution, pooling and image layouts
- SVG serialization, plotting and display error handling
- Input validation and error handling

### Building and Running Tests

Enable unit tests during CMake configuration:

```bash
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DNEURO_BUILD_TESTS=ON
cmake --build build -j
```

GoogleTest is automatically downloaded and configured through CMake's FetchContent module.

Run the complete test suite:

```bash
ctest --test-dir build --output-on-failure
```

To run a specific group of tests, use a regular expression:

```bash
ctest --test-dir build -R TrainerTest --output-on-failure
```

### Test Status

The full test suite requires both the MNIST and CIFAR-10 datasets described above. CPU testing does not require a GPU, and visualization unit tests do not require a graphical environment.

To run the tests on CPU:

```bash
cmake -S . -B build/cpu -DNEURO_BACKEND=CPU -DNEURO_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpu -j
ctest --test-dir build/cpu --output-on-failure --no-tests=error
```

### Continuous Integration

The GitHub Actions workflow is configured to compile the CUDA backend and build and execute the CPU test suite. The CPU Dockerfile downloads the required MNIST and CIFAR-10 files automatically. CUDA runtime tests are performed locally.

## Current Limitations

Neuro is an early-stage deep learning library and is not intended to replace established frameworks such as PyTorch or TensorFlow.

The current release has the following limitations:

- **Platform support** — Currently developed and tested on Linux with CPU and NVIDIA CUDA backends.
- **Neural network architectures** — Fully connected and convolutional networks are supported. Recurrent and attention-based layers are not yet supported.
- **GPU computation** — GPU operations rely on ArrayFire rather than custom CUDA kernels.
- **Model persistence** — Model serialization and training checkpointing are not yet supported.
- **Training features** — Early stopping retains the model from the stopping epoch; automatic restoration of the best model and general training callbacks are not yet supported.
- **Visualization** — SVG line plots and Linux display are supported; native tensor image and feature map visualization are not yet implemented.
- **Performance** — No formal performance benchmarks against established deep learning frameworks have been conducted.
- **API stability** — The public API is experimental and may change between minor releases.

## Roadmap

Neuro is under active development, with future releases focusing on expanding neural network capabilities, improving performance and extending the training API.

### Neural Network Components

- [x] Convolutional layers
- [x] Pooling layers
- [x] Dropout
- [x] Batch normalization
- [ ] Custom module composition and registration
- [ ] Residual networks
- [ ] Additional activation and loss functions
- [ ] Recurrent layers
- [ ] Attention mechanisms

### Training and Optimization

- [x] Training and evaluation modes
- [x] Per-epoch validation
- [x] Dataset transforms and augmentation
- [x] Learning rate schedulers
- [x] Early stopping
- [ ] Model serialization
- [ ] Training checkpoints
- [x] AdamW and weight decay
- [ ] Additional optimizers
- [ ] Training callbacks

### Tensor and Autograd Engine

- [ ] Extended tensor operations
- [ ] Improved memory management
- [ ] Computational graph optimizations
- [ ] Reduced CPU-GPU synchronization
- [ ] Performance profiling and benchmarking

### Visualization

- [x] SVG plotting and training history integration
- [ ] Image and feature map visualization
- [ ] Model inspection and activation visualization

### Developer Experience

- [ ] Comprehensive API documentation
- [x] Continuous integration
- [ ] Additional examples and tutorials
- [ ] Extended platform support

The next development milestone is model architecture and persistence, including custom module composition, recursive parameter and buffer registration, residual networks, model serialization and training checkpoints.

## License

Neuro is licensed under the MIT License.

See the [LICENSE](LICENSE) file for details.
