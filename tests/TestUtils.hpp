
#pragma once

#include <arrayfire.h>
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

namespace neuro::testing {

inline void expectArrayNear(const af::array& actual, const af::dim4& expectedShape,
                            const std::vector<float>& expected, float tolerance = 1e-5f) {
    ASSERT_TRUE(actual.dims() == expectedShape) << "Tensor shape mismatch";

    ASSERT_EQ(actual.elements(), expected.size()) << "Tensor element count mismatch";

    std::vector<float> values(expected.size());
    actual.as(f32).host(values.data());

    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(values[i], expected[i], tolerance) << "Mismatch at flat index " << i;
    }
}

} // namespace neuro::testing
