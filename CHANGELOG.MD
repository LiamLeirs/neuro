# Changelog

All notable changes to Neuro will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.3.0] - 2026-11-28

Training utilities, regularization and improved generalization.

### Added

- Differentiable tensor mean reductions, variance and square root operations.
- Differentiable dropout, padding, slicing, cropping, flipping and conditional selection.
- Optimized batched tensor cropping with independent offsets per sample.
- Dropout, BatchNorm1D and BatchNorm2D neural network modules.
- Training and evaluation modes with recursive propagation through sequential models.
- Weight decay support for SGD and Adam.
- AdamW optimizer with decoupled weight decay.
- Configurable optimizer learning rates.
- StepLR, ExponentialLR, CosineAnnealingLR and ReduceLROnPlateau learning rate schedulers.
- Dataset transform abstraction and composable transformation pipelines.
- Normalize, RandomHorizontalFlip and RandomCrop transforms.
- TransformDataset with separate input and target transformations.
- Optimized batch transformations with per-sample fallback.
- Direct construction of TransformDataset using concrete transforms.
- Per-epoch validation with separate training and validation metrics.
- Learning rate tracking in training history.
- EarlyStopping with configurable patience, improvement threshold and metric monitoring.
- Trainer integration for learning rate schedulers and early stopping.
- Unit tests for regularization, augmentation, optimizers, schedulers and training utilities.

### Changed

- Trainer automatically switches between training and evaluation modes.
- Updated the CIFAR-10 example with batch normalization, dropout, data augmentation, AdamW and cosine learning rate scheduling.
- Introduced a separate validation split in the CIFAR-10 example.
- Improved CIFAR-10 test accuracy from 73.24% to 84.09% in local training runs.
- Extended CIFAR-10 visualization with validation metrics and learning rate curves.
- Expanded documentation to cover the new training and regularization features.

## [0.2.0] - 2026-11-26

Convolutional neural networks and basic visualization.

### Added

- Differentiable tensor reshaping, dimension reordering and flattening.
- Differentiable 2D convolution, max pooling, average pooling and global average pooling.
- Conv2D, MaxPool2D, AvgPool2D, GlobalAvgPool2D and Flatten neural network modules.
- MNIST image layouts for convolutional networks, alongside the existing flattened layout.
- CIFAR10Dataset with optimized NCHW batch loading.
- MNIST CNN and CIFAR-10 CNN examples.
- Custom SVG document renderer with drawing primitives and rotated text.
- Line plotting with automatic axis scaling, ticks, grid lines, colors, titles, axis labels and legends.
- Direct plotting of TrainingHistory metrics and arbitrary numerical series.
- SVG export, plot reset and Linux display through xdg-open.
- Unit tests for convolution, pooling, tensor shape operations, image datasets and visualization.
- Automatic CIFAR-10 dataset setup in the CPU Docker environment.

### Changed

- Extended example applications and documentation to cover convolutional networks and visualization.

## [0.1.0] - 2026-11-25

Initial public release of Neuro.

### Added

- GPU-accelerated tensor operations using ArrayFire and CUDA.
- CPU backend support through ArrayFire.
- Configurable CPU and CUDA backends through the NEURO_BACKEND CMake option.
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
- 387 unit tests passing in Release mode on both CPU and CUDA backends.
- Reproducible Docker environments for CPU testing and CUDA build verification.
- Automatic MNIST dataset setup in the CPU Docker environment.
- GitHub Actions workflows for CUDA build verification and automated CPU testing.
- Docker build caching for continuous integration.
- Project documentation and MIT license.
