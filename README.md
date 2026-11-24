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

- **GPU-accelerated tensors** — Tensor arithmetic, broadcasting, matrix multiplication, reductions and indexing powered by ArrayFire.
- **Configurable backend** — CUDA acceleration by default, with CPU support selected at build time.
- **Automatic differentiation** — Dynamic computational graphs, reverse-mode gradient propagation and gradient tracking control.
- **Neural networks** — Fully connected layers, activation functions and composable sequential models.
- **Optimizers** — Stochastic Gradient Descent (SGD) and Adam.
- **Loss functions** — Mean Squared Error (MSE), Mean Absolute Error (MAE) and Cross-Entropy Loss.
- **Data utilities** — Dataset abstractions, batched data loading, shuffling and optimized MNIST data loading.
- **Training utilities** — Reusable training and evaluation loops, extensible metrics and training history.
- **CMake integration** — Installable CMake package with support for `find_package(Neuro)`.
- **Unit testing** — 387 tests passing locally on both CPU and CUDA, covering tensor operations, automatic differentiation, neural network components and training utilities.

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
cmake -S . -B build \\
    -DCMAKE_BUILD_TYPE=Release \\
    -DNEURO_BUILD_EXAMPLES=ON \\
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
cmake -S . -B build \\
    -DCMAKE_BUILD_TYPE=Release \\
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
find_package(Neuro 0.1 CONFIG REQUIRED)
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
| MNIST Classification | Train and evaluate a neural network on handwritten digit recognition. |

### Running the Examples

Build Neuro with examples enabled:

```bash
cmake -S . -B build \\
    -DCMAKE_BUILD_TYPE=Release \\
    -DNEURO_BUILD_EXAMPLES=ON
cmake --build build -j
```

Run the examples from the repository root:

```bash
./build/examples/neuro_linear_regression
./build/examples/neuro_xor
./build/examples/neuro_mnist
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

## Testing

Neuro includes a comprehensive unit test suite built with GoogleTest. The tests cover:

- Tensor operations and broadcasting
- Automatic differentiation and gradient accumulation
- Gradient tracking control
- Neural network layers and activation functions
- Loss functions and optimizers
- Datasets and data loading
- Metrics and training utilities
- Input validation and error handling

### Building and Running Tests

Enable unit tests during CMake configuration:

```bash
cmake -S . -B build \\
    -DCMAKE_BUILD_TYPE=Release \\
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

As of v0.1.0, all 387 unit tests pass in Release mode.

All 387 tests have passed locally on both CPU and CUDA in Release mode. CPU testing does not require a GPU.

The full test suite requires the MNIST files described in [Dataset Setup](#dataset-setup). To run the tests on CPU:

```bash
cmake -S . -B build/cpu -DNEURO_BACKEND=CPU -DNEURO_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpu -j
ctest --test-dir build/cpu --output-on-failure --no-tests=error
```

### Continuous Integration

The GitHub Actions workflow is configured to compile the CUDA backend and build and execute the CPU test suite. The CPU Dockerfile downloads the required MNIST files automatically. CUDA runtime tests are performed locally.

## Current Limitations

Neuro is an early-stage deep learning library and is not intended to replace established frameworks such as PyTorch or TensorFlow.

The current release has the following limitations:

- **Platform support** — Currently developed and tested on Linux with CPU and NVIDIA CUDA backends.
- **Neural network architectures** — Limited to fully connected networks. Convolutional, recurrent and attention-based layers are not yet supported.
- **GPU computation** — GPU operations rely on ArrayFire rather than custom CUDA kernels.
- **Model persistence** — Model serialization and training checkpointing are not yet supported.
- **Training features** — Advanced features such as learning rate scheduling, dropout and batch normalization are not yet implemented.
- **Performance** — No formal performance benchmarks against established deep learning frameworks have been conducted.
- **API stability** — The public API is experimental and may change between minor releases.

## Roadmap

Neuro is under active development, with future releases focusing on expanding neural network capabilities, improving performance and extending the training API.

### Neural Network Components

- [ ] Convolutional layers
- [ ] Pooling layers
- [ ] Dropout
- [ ] Batch normalization
- [ ] Additional activation and loss functions
- [ ] Recurrent layers
- [ ] Attention mechanisms

### Training and Optimization

- [ ] Learning rate schedulers
- [ ] Model serialization
- [ ] Training checkpoints
- [ ] Additional optimizers
- [ ] Training callbacks

### Tensor and Autograd Engine

- [ ] Extended tensor operations
- [ ] Improved memory management
- [ ] Computational graph optimizations
- [ ] Reduced CPU-GPU synchronization
- [ ] Performance profiling and benchmarking

### Developer Experience

- [ ] Comprehensive API documentation
- [x] Continuous integration
- [ ] Additional examples and tutorials
- [ ] Extended platform support

The next major development milestone is support for convolutional

neural networks (CNNs).

## License

Neuro is licensed under the MIT License.

See the [LICENSE](LICENSE) file for details.