#include "TestUtils.hpp"
#include "neuro/GradMode.hpp"
#include "neuro/Tensor.hpp"

#include <gtest/gtest.h>
#include <stdexcept>
#include <thread>
#include <type_traits>

using namespace neuro;
using neuro::testing::expectArrayNear;

class GradModeTest : public ::testing::Test {
protected:
    void SetUp() override {
        GradMode::setEnabled(true);
    }

    void TearDown() override {
        GradMode::setEnabled(true);
    }
};

// ==================== GradMode ====================

TEST_F(GradModeTest, EnabledByDefaultInNewThread) {
    bool enabled = false;

    std::thread worker([&]() { enabled = GradMode::isEnabled(); });

    worker.join();

    EXPECT_TRUE(enabled);
}

TEST_F(GradModeTest, ToggleGradientMode) {
    GradMode::setEnabled(false);
    EXPECT_FALSE(GradMode::isEnabled());

    GradMode::setEnabled(true);
    EXPECT_TRUE(GradMode::isEnabled());
}

TEST_F(GradModeTest, ThreadLocalState) {
    GradMode::setEnabled(false);

    std::thread worker([]() {
        EXPECT_TRUE(GradMode::isEnabled());

        GradMode::setEnabled(false);
        EXPECT_FALSE(GradMode::isEnabled());

        GradMode::setEnabled(true);
    });

    worker.join();

    EXPECT_FALSE(GradMode::isEnabled());
}

// ==================== Tensor Integration ====================

TEST_F(GradModeTest, EnabledTracksGradients) {
    Tensor x({1, 2, 3}, true);
    Tensor y = x * 2.0;

    EXPECT_TRUE(y.requires_grad());
}

TEST_F(GradModeTest, DisabledPreventsGradientTracking) {
    Tensor x({1, 2, 3}, true);

    GradMode::setEnabled(false);

    Tensor y = x * 2.0;

    EXPECT_FALSE(y.requires_grad());
}

TEST_F(GradModeTest, PreservesExistingTensorState) {
    Tensor x({1, 2, 3}, true);
    Tensor y = x * 2.0;

    GradMode::setEnabled(false);

    EXPECT_TRUE(x.requires_grad());
    EXPECT_TRUE(y.requires_grad());
}

TEST_F(GradModeTest, ForwardComputationUnaffected) {
    Tensor x({1, 2, 3}, true);

    GradMode::setEnabled(false);

    Tensor y = x * 2.0 + 1.0;

    expectArrayNear(y.data(), af::dim4(3), {3, 5, 7});
    EXPECT_FALSE(y.requires_grad());
}

TEST_F(GradModeTest, ReenabledRestoresAutograd) {
    Tensor x({1, 2, 3}, true);

    GradMode::setEnabled(false);
    Tensor y = x * 2.0;

    GradMode::setEnabled(true);
    Tensor z = x * 3.0;

    EXPECT_FALSE(y.requires_grad());
    EXPECT_TRUE(z.requires_grad());

    z.sum().backward();

    expectArrayNear(x.grad(), af::dim4(3), {3, 3, 3});
}

TEST_F(GradModeTest, DisabledResultsRemainDetached) {
    Tensor x({1, 2, 3}, true);

    GradMode::setEnabled(false);
    Tensor y = x * 2.0;

    GradMode::setEnabled(true);
    Tensor z = y * 3.0;

    EXPECT_FALSE(y.requires_grad());
    EXPECT_FALSE(z.requires_grad());
}

TEST_F(GradModeTest, EnabledDoesNotForceGradientTracking) {
    Tensor x({1, 2, 3}, false);
    Tensor y = x * 2.0;

    EXPECT_FALSE(y.requires_grad());
}

// ==================== NoGradGuard ====================

TEST_F(GradModeTest, GuardDisablesGradients) {
    {
        NoGradGuard guard;
        EXPECT_FALSE(GradMode::isEnabled());
    }

    EXPECT_TRUE(GradMode::isEnabled());
}

TEST_F(GradModeTest, GuardRestoresDisabledState) {
    GradMode::setEnabled(false);

    {
        NoGradGuard guard;
        EXPECT_FALSE(GradMode::isEnabled());
    }

    EXPECT_FALSE(GradMode::isEnabled());
}

TEST_F(GradModeTest, NestedGuards) {
    EXPECT_TRUE(GradMode::isEnabled());

    {
        NoGradGuard outer;
        EXPECT_FALSE(GradMode::isEnabled());

        {
            NoGradGuard inner;
            EXPECT_FALSE(GradMode::isEnabled());
        }

        EXPECT_FALSE(GradMode::isEnabled());
    }

    EXPECT_TRUE(GradMode::isEnabled());
}

TEST_F(GradModeTest, GuardExceptionSafety) {
    EXPECT_THROW(
        {
            NoGradGuard guard;
            throw std::runtime_error("Test exception");
        },
        std::runtime_error);

    EXPECT_TRUE(GradMode::isEnabled());
}

TEST_F(GradModeTest, GuardTensorIntegration) {
    Tensor x({1, 2, 3}, true);

    {
        NoGradGuard guard;

        Tensor y = x * 2.0;

        EXPECT_FALSE(y.requires_grad());
        EXPECT_TRUE(x.requires_grad());

        expectArrayNear(y.data(), af::dim4(3), {2, 4, 6});
    }

    Tensor z = x * 3.0;

    EXPECT_TRUE(z.requires_grad());

    z.sum().backward();

    expectArrayNear(x.grad(), af::dim4(3), {3, 3, 3});
}

TEST_F(GradModeTest, GuardIsNonCopyableAndNonMovable) {
    EXPECT_FALSE(std::is_copy_constructible_v<NoGradGuard>);
    EXPECT_FALSE(std::is_copy_assignable_v<NoGradGuard>);
    EXPECT_FALSE(std::is_move_constructible_v<NoGradGuard>);
    EXPECT_FALSE(std::is_move_assignable_v<NoGradGuard>);
}