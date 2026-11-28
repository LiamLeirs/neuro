#include "neuro/training/EarlyStopping.hpp"

#include <gtest/gtest.h>
#include <limits>

using namespace neuro;

TEST(EarlyStoppingTest, FirstObservationEstablishesBaseline) {
    EarlyStopping stopping(PlateauMode::Min, 0);

    stopping.step(1.0);

    EXPECT_FALSE(stopping.shouldStop());
}

TEST(EarlyStoppingTest, StopsAfterPatienceExceeded) {
    EarlyStopping stopping(PlateauMode::Min, 3);

    stopping.step(1.0);

    for (int i = 0; i < 2; ++i) {
        stopping.step(1.1);
        EXPECT_FALSE(stopping.shouldStop());
    }

    stopping.step(1.1);
    EXPECT_TRUE(stopping.shouldStop());
}

TEST(EarlyStoppingTest, ZeroAndOnePatience) {
    for (std::size_t patience : {0, 1}) {
        EarlyStopping stopping(PlateauMode::Min, patience);

        stopping.step(1.0);
        EXPECT_FALSE(stopping.shouldStop());

        stopping.step(1.1);
        EXPECT_TRUE(stopping.shouldStop());
    }
}

TEST(EarlyStoppingTest, ImprovementResetsPatience) {
    EarlyStopping stopping(PlateauMode::Min, 2);

    stopping.step(1.0);
    stopping.step(1.1);
    stopping.step(0.8);
    stopping.step(0.9);

    EXPECT_FALSE(stopping.shouldStop());

    stopping.step(0.9);
    EXPECT_TRUE(stopping.shouldStop());
}

TEST(EarlyStoppingTest, SupportsMaximization) {
    EarlyStopping stopping(PlateauMode::Max, 2);

    stopping.step(0.5);
    stopping.step(0.7);
    stopping.step(0.6);

    EXPECT_FALSE(stopping.shouldStop());

    stopping.step(0.6);
    EXPECT_TRUE(stopping.shouldStop());
}

TEST(EarlyStoppingTest, RequiresSufficientImprovement) {
    EarlyStopping stopping(PlateauMode::Min, 2, 0.125);

    stopping.step(1.0);
    stopping.step(0.9375);
    stopping.step(0.875);

    EXPECT_TRUE(stopping.shouldStop());
}

TEST(EarlyStoppingTest, ComparesAgainstHistoricalBest) {
    EarlyStopping stopping(PlateauMode::Min, 2);

    stopping.step(0.5);
    stopping.step(0.8);
    stopping.step(0.7);

    EXPECT_TRUE(stopping.shouldStop());
}

TEST(EarlyStoppingTest, RemainsStoppedAfterImprovement) {
    EarlyStopping stopping(PlateauMode::Min, 1);

    stopping.step(1.0);
    stopping.step(1.1);

    ASSERT_TRUE(stopping.shouldStop());

    stopping.step(0.5);

    EXPECT_TRUE(stopping.shouldStop());
}

TEST(EarlyStoppingTest, ResetClearsState) {
    EarlyStopping stopping(PlateauMode::Min, 1);

    stopping.step(0.1);
    stopping.step(0.2);

    ASSERT_TRUE(stopping.shouldStop());

    stopping.reset();

    EXPECT_FALSE(stopping.shouldStop());

    stopping.step(2.0);
    EXPECT_FALSE(stopping.shouldStop());

    stopping.step(2.1);
    EXPECT_TRUE(stopping.shouldStop());
}

TEST(EarlyStoppingTest, RejectsInvalidConfiguration) {
    EXPECT_THROW(EarlyStopping(static_cast<PlateauMode>(99)), std::invalid_argument);

    for (double threshold : {-0.1, std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::quiet_NaN()}) {
        EXPECT_THROW(EarlyStopping(PlateauMode::Min, 3, threshold), std::invalid_argument);
    }
}

TEST(EarlyStoppingTest, RejectsNonFiniteMetricsWithoutChangingState) {
    EarlyStopping stopping(PlateauMode::Min, 2);

    stopping.step(1.0);

    for (double metric :
         {std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
          std::numeric_limits<double>::quiet_NaN()}) {
        EXPECT_THROW(stopping.step(metric), std::invalid_argument);
    }

    stopping.step(1.1);
    EXPECT_FALSE(stopping.shouldStop());

    stopping.step(1.1);
    EXPECT_TRUE(stopping.shouldStop());
}