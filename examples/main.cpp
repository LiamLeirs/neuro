
#include <arrayfire.h>
#include <iostream>

int main() {
    try {
        af::info();

        af::array a = af::randu(3, 3);
        af::array b = af::randu(3, 3);

        af::array c = af::matmul(a, b);

        af_print(a);
        af_print(b);
        af_print(c);

        af::sync();

        return 0;
    }
    catch (const af::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
