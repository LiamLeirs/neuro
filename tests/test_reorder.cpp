
#include "neuro/GradMode.hpp"
#include "neuro/Tensor.hpp"

#include <af/data.h>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

using namespace neuro;

namespace {

void expectValues(const af::array& array, const std::vector<float>& expected) {
    ASSERT_EQ(array.elements(), expected.size());

    std::vector<float> actual(array.elements());
    array.as(f32).host(actual.data());

    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], 1e-5f) << "Mismatch at linear index " << i;
    }
}

} // namespace

// ========================================
// Forward propagation
// ========================================

TEST(ReorderTest, IdentityPermutation) {
    Tensor x({{1, 2, 3}, {4, 5, 6}});

    Tensor y = x.reorder(0, 1, 2, 3);

    EXPECT_EQ(y.shape(), x.shape());
    expectValues(y.data(), {1, 4, 2, 5, 3, 6});
}

TEST(ReorderTest, Transpose2D) {
    Tensor x({{1, 2, 3}, {4, 5, 6}});

    Tensor y = x.reorder(1, 0, 2, 3);

    EXPECT_EQ(y.shape(), af::dim4(3, 2));
    expectValues(y.data(), {1, 2, 3, 4, 5, 6});
}

TEST(ReorderTest, MoveVectorToSecondDimension) {
    Tensor x({1, 2, 3, 4});

    Tensor y = x.reorder(1, 0, 2, 3);

    EXPECT_EQ(y.shape(), af::dim4(1, 4));
    expectValues(y.data(), {1, 2, 3, 4});
}

TEST(ReorderTest, MoveVectorToFourthDimension) {
    Tensor x({1, 2, 3, 4});

    Tensor y = x.reorder(1, 2, 3, 0);

    EXPECT_EQ(y.shape(), af::dim4(1, 1, 1, 4));
    expectValues(y.data(), {1, 2, 3, 4});
}

TEST(ReorderTest, Permute3D) {
    Tensor x(af::iota(af::dim4(2, 2, 3)));

    Tensor y = x.reorder(2, 0, 1, 3);

    EXPECT_EQ(y.shape(), af::dim4(3, 2, 2));

    expectValues(y.data(), {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11});
}

TEST(ReorderTest, Permute4D) {
    Tensor x(af::iota(af::dim4(2, 2, 2, 2)));

    Tensor y = x.reorder(3, 0, 1, 2);

    EXPECT_EQ(y.shape(), af::dim4(2, 2, 2, 2));

    expectValues(y.data(), {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15});
}

TEST(ReorderTest, PermuteUnequalDimensions) {
    Tensor x(af::iota(af::dim4(2, 3, 4, 5)));

    Tensor y = x.reorder(3, 0, 2, 1);

    EXPECT_EQ(y.shape(), af::dim4(5, 2, 4, 3));
    EXPECT_EQ(y.data().elements(), 120);
}

TEST(ReorderTest, ForwardAndInverseRestoreTensor) {
    Tensor x(af::iota(af::dim4(2, 2, 2, 2)));

    Tensor y = x.reorder(3, 0, 1, 2);
    Tensor z = y.reorder(1, 2, 3, 0);

    EXPECT_EQ(z.shape(), x.shape());

    expectValues(z.data(), {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
}

TEST(ReorderTest, PreservesDataType) {
    Tensor x(af::iota(af::dim4(2, 3)).as(s32));

    Tensor y = x.reorder(1, 0, 2, 3);

    EXPECT_EQ(y.data().type(), s32);
    expectValues(y.data(), {0, 2, 4, 1, 3, 5});
}

TEST(ReorderTest, DoesNotModifyOriginalTensor) {
    Tensor x({{1, 2, 3}, {4, 5, 6}});

    Tensor y = x.reorder(1, 0, 2, 3);

    EXPECT_EQ(x.shape(), af::dim4(2, 3));
    EXPECT_EQ(y.shape(), af::dim4(3, 2));

    expectValues(x.data(), {1, 4, 2, 5, 3, 6});
}

TEST(ReorderTest, AllValidPermutations) {
    Tensor x(af::iota(af::dim4(2, 3, 4, 5)));

    int count = 0;

    for (int a = 0; a < 4; ++a) {
        for (int b = 0; b < 4; ++b) {
            for (int c = 0; c < 4; ++c) {
                for (int d = 0; d < 4; ++d) {
                    if (a == b || a == c || a == d || b == c || b == d || c == d) {
                        continue;
                    }

                    Tensor y = x.reorder(a, b, c, d);

                    EXPECT_EQ(y.shape(),
                              af::dim4(x.shape()[a], x.shape()[b], x.shape()[c], x.shape()[d]));

                    EXPECT_EQ(y.data().elements(), 120);
                    ++count;
                }
            }
        }
    }

    EXPECT_EQ(count, 24);
}

// ========================================
// Input validation
// ========================================

TEST(ReorderTest, RejectsNegativeDimension) {
    Tensor x({1, 2, 3, 4});

    EXPECT_THROW(x.reorder(-1, 0, 1, 2), std::invalid_argument);
}

TEST(ReorderTest, RejectsDimensionAboveThree) {
    Tensor x({1, 2, 3, 4});

    EXPECT_THROW(x.reorder(0, 1, 2, 4), std::invalid_argument);
}

TEST(ReorderTest, RejectsDuplicateDimension) {
    Tensor x({1, 2, 3, 4});

    EXPECT_THROW(x.reorder(0, 0, 1, 2), std::invalid_argument);
}

TEST(ReorderTest, RejectsNonAdjacentDuplicate) {
    Tensor x({1, 2, 3, 4});

    EXPECT_THROW(x.reorder(0, 1, 2, 0), std::invalid_argument);
}

TEST(ReorderTest, RejectsAllIdenticalDimensions) {
    Tensor x({1, 2, 3, 4});

    EXPECT_THROW(x.reorder(1, 1, 1, 1), std::invalid_argument);
}

TEST(ReorderTest, RejectsInvalidLastDimension) {
    Tensor x({1, 2, 3, 4});

    EXPECT_THROW(x.reorder(0, 1, 2, -1), std::invalid_argument);
}

// ========================================
// Automatic differentiation
// ========================================

TEST(ReorderTest, PreservesRequiresGrad) {
    Tensor x({{1, 2}, {3, 4}}, true);

    Tensor y = x.reorder(1, 0, 2, 3);

    EXPECT_TRUE(y.requires_grad());
}

TEST(ReorderTest, BackwardIdentityPermutation) {
    Tensor x({1, 2, 3, 4}, true);

    Tensor y = x.reorder(0, 1, 2, 3);
    Tensor weights({2, 4, 6, 8});

    (y * weights).sum().backward();

    EXPECT_EQ(x.grad().dims(), x.shape());
    expectValues(x.grad(), {2, 4, 6, 8});
}

TEST(ReorderTest, BackwardTranspose) {
    Tensor x({{1, 2, 3}, {4, 5, 6}}, true);

    Tensor y = x.reorder(1, 0, 2, 3);

    Tensor weights({{10, 20}, {30, 40}, {50, 60}});

    (y * weights).sum().backward();

    EXPECT_EQ(x.grad().dims(), af::dim4(2, 3));

    expectValues(x.grad(), {10, 20, 30, 40, 50, 60});
}

TEST(ReorderTest, BackwardNonSymmetric3DPermutation) {
    Tensor x(af::iota(af::dim4(2, 2, 3)), true);

    Tensor y = x.reorder(2, 0, 1, 3);

    Tensor weights(af::iota(af::dim4(3, 2, 2)) + 1);

    (y * weights).sum().backward();

    EXPECT_EQ(x.grad().dims(), af::dim4(2, 2, 3));

    expectValues(x.grad(), {1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12});
}

TEST(ReorderTest, BackwardNonSymmetric4DPermutation) {
    Tensor x(af::iota(af::dim4(2, 2, 2, 2)), true);

    Tensor y = x.reorder(3, 0, 1, 2);

    Tensor weights(af::iota(af::dim4(2, 2, 2, 2)) + 1);

    (y * weights).sum().backward();

    EXPECT_EQ(x.grad().dims(), x.shape());

    expectValues(x.grad(), {1, 3, 5, 7, 9, 11, 13, 15, 2, 4, 6, 8, 10, 12, 14, 16});
}

TEST(ReorderTest, BackwardRestoresUnequalDimensions) {
    Tensor x(af::iota(af::dim4(2, 3, 4, 5)), true);

    Tensor y = x.reorder(3, 0, 2, 1);

    y.sum().backward();

    EXPECT_EQ(x.grad().dims(), af::dim4(2, 3, 4, 5));
    expectValues(x.grad(), std::vector<float>(120, 1.0f));
}

TEST(ReorderTest, BackwardThroughChainedPermutations) {
    Tensor x(af::iota(af::dim4(2, 2, 2, 2)), true);

    Tensor y = x.reorder(3, 0, 1, 2);
    Tensor z = y.reorder(1, 2, 3, 0);

    Tensor weights(af::iota(af::dim4(2, 2, 2, 2)) + 1);

    (z * weights).sum().backward();

    EXPECT_EQ(x.grad().dims(), x.shape());

    expectValues(x.grad(), {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
}

TEST(ReorderTest, BackwardThroughMultipleBranches) {
    Tensor x(af::iota(af::dim4(2, 2, 3)), true);

    Tensor a = x.reorder(2, 0, 1, 3);
    Tensor b = x.reorder(1, 2, 0, 3);

    Tensor loss = (a * 2.0f).sum() + (b * 3.0f).sum();

    loss.backward();

    EXPECT_EQ(x.grad().dims(), x.shape());
    expectValues(x.grad(), std::vector<float>(12, 5.0f));
}

TEST(ReorderTest, BackwardWithRepeatedParent) {
    Tensor x({{1, 2}, {3, 4}}, true);

    Tensor y = x.reorder(1, 0, 2, 3);

    (y * y).sum().backward();

    expectValues(x.grad(), {2, 6, 4, 8});
}

TEST(ReorderTest, BackwardThroughPreviousOperation) {
    Tensor x({{1, 2}, {3, 4}}, true);

    Tensor y = x * x;
    Tensor z = y.reorder(1, 0, 2, 3);

    z.sum().backward();

    expectValues(x.grad(), {2, 6, 4, 8});
}

TEST(ReorderTest, BackwardCombinedWithReshape) {
    Tensor x(af::iota(af::dim4(2, 2, 2, 2)), true);

    Tensor y = x.reorder(3, 0, 1, 2);
    Tensor z = y.reshape(af::dim4(2, 8));

    Tensor weights(af::iota(af::dim4(2, 8)) + 1);

    (z * weights).sum().backward();

    EXPECT_EQ(x.grad().dims(), x.shape());

    expectValues(x.grad(), {1, 3, 5, 7, 9, 11, 13, 15, 2, 4, 6, 8, 10, 12, 14, 16});
}

// ========================================
// Gradient tracking
// ========================================

TEST(ReorderTest, DetachedTensorRemainsDetached) {
    Tensor x({{1, 2}, {3, 4}}, false);

    Tensor y = x.reorder(1, 0, 2, 3);

    EXPECT_FALSE(y.requires_grad());
}

TEST(ReorderTest, RespectsNoGradGuard) {
    Tensor x({{1, 2, 3}, {4, 5, 6}}, true);

    NoGradGuard guard;

    Tensor y = x.reorder(1, 0, 2, 3);

    EXPECT_FALSE(y.requires_grad());
    EXPECT_TRUE(x.requires_grad());

    EXPECT_EQ(y.shape(), af::dim4(3, 2));
    expectValues(y.data(), {1, 2, 3, 4, 5, 6});
}

TEST(ReorderTest, GradientTrackingRestoredAfterGuard) {
    Tensor x({{1, 2}, {3, 4}}, true);

    {
        NoGradGuard guard;

        Tensor y = x.reorder(1, 0, 2, 3);

        EXPECT_FALSE(y.requires_grad());
    }

    Tensor z = x.reorder(1, 0, 2, 3);

    EXPECT_TRUE(z.requires_grad());

    z.sum().backward();

    expectValues(x.grad(), {1, 1, 1, 1});
}
