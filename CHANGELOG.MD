# Changelog

All notable changes to Neuro will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0]

Initial public release of Neuro.

### Added

- GPU-accelerated tensor operations using ArrayFire and CUDA.
- Reverse-mode automatic differentiation with dynamic computational graphs.
- Broadcasting, matrix multiplication, reductions, indexing and stacking.
- Gradient tracking control through GradMode and NoGradGuard.
- Neural network modules, linear layers and sequential models.
- ReLU, LeakyReLU, Sigmoid, Tanh, Softmax and LogSoftmax activations.
- SGD and Adam optimizers.
- MSE, MAE and Cross-Entropy loss functions.
- Dataset abstractions, TensorDataset and MNISTDataset.
- Batched data loading with optional shuffling.
- Optimized MNIST batch loading.
- Trainer API with training, evaluation and metric registration.
- Accuracy and weighted mean metrics.
- Linear regression, XOR and MNIST examples.
- Installable CMake package with version compatibility checks.
- 387 unit tests passing in Release mode.
- Project documentation and MIT license.