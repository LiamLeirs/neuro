
#include "neuro/Tensor.hpp"
#include "neuro/data/MNISTDataset.hpp"

#include <arrayfire.h>
#include <iostream>
#include <ostream>

using namespace neuro;

int main() {
    MNISTDataset dataset("data/mnist/train-images.idx3-ubyte",
                         "data/mnist/train-labels.idx1-ubyte");
    std::cout << dataset.size() << std::endl;
    std::cout << dataset.get(0).input.shape() << std::endl;
}
