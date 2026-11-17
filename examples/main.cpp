
#include "neuro/Tensor.hpp"

#include <arrayfire.h>

int main() {
    af::array a = af::randu(3, 3);
    neuro::Tensor t(a);
    return 0;
}
