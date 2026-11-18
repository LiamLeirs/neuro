
#include "neuro/Tensor.hpp"

#include <af/array.h>
#include <af/random.h>
#include <arrayfire.h>
#include <iostream>

int main() {
    af::array a = af::randu(3, 3);
    af::array b = af::randu(3, 3);
    neuro::Tensor t1(a);
    neuro::Tensor t2(b);
    neuro::Tensor t3 = t1 * t2;
    std::cout << 5 * t3 * 2;
    return 0;
}
