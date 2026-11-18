
#include "neuro/Tensor.hpp"

#include <af/array.h>
#include <af/random.h>
#include <arrayfire.h>
#include <iostream>

int main() {
    neuro::Tensor t = {{1, 2, 3}, {4, 5, 6}};
    std::cout << t.transpose();
    return 0;
}
