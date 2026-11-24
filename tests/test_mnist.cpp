
#include "neuro/data/DataLoader.hpp"
#include "neuro/data/MNISTDataset.hpp"

#include <arrayfire.h>
#include <cstddef>
#include <filesystem>
#include <gtest/gtest.h>
#include <stdexcept>

using namespace neuro;

namespace {

const std::filesystem::path MNIST_DIR =
    std::filesystem::path(__FILE__).parent_path().parent_path() / "data/mnist";

MNISTDataset loadTrainingData() {
    return MNISTDataset((MNIST_DIR / "train-images.idx3-ubyte").string(),
                        (MNIST_DIR / "train-labels.idx1-ubyte").string());
}

MNISTDataset loadTestData() {
    return MNISTDataset((MNIST_DIR / "t10k-images.idx3-ubyte").string(),
                        (MNIST_DIR / "t10k-labels.idx1-ubyte").string());
}

class MNISTTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        dataset = std::make_unique<MNISTDataset>((MNIST_DIR / "train-images.idx3-ubyte").string(),
                                                 (MNIST_DIR / "train-labels.idx1-ubyte").string());
    }

    static void TearDownTestSuite() {
        dataset.reset();
    }

    static std::unique_ptr<MNISTDataset> dataset;
};

std::unique_ptr<MNISTDataset> MNISTTest::dataset;

} // namespace

// ==================== Dataset Loading ====================

TEST_F(MNISTTest, TrainingDatasetSize) {
    EXPECT_EQ(dataset->size(), 60000);
}

TEST(MNISTDatasetTest, TestDatasetSize) {
    auto dataset = loadTestData();

    EXPECT_EQ(dataset.size(), 10000);
}

// ==================== Sample Shapes ====================

TEST_F(MNISTTest, InputShape) {
    auto sample = dataset->get(0);

    EXPECT_EQ(sample.input.shape(), af::dim4(784));
}

TEST_F(MNISTTest, TargetShape) {
    auto sample = dataset->get(0);

    EXPECT_EQ(sample.target.data().elements(), 1);
}

// ==================== Data Validation ====================

TEST_F(MNISTTest, PixelsAreNormalized) {
    // Check several samples across the dataset.
    for (std::size_t i : {0u, 100u, 1000u, 30000u, 59999u}) {
        auto sample = dataset->get(i);

        af::array image = sample.input.data();

        EXPECT_EQ(image.type(), f32);
        EXPECT_GE(af::min<float>(image), 0.0f);
        EXPECT_LE(af::max<float>(image), 1.0f);
    }
}

TEST_F(MNISTTest, LabelsAreValid) {
    for (std::size_t i : {0u, 100u, 1000u, 30000u, 59999u}) {
        auto sample = dataset->get(i);

        af::array label = sample.target.data();

        EXPECT_EQ(label.elements(), 1);
        EXPECT_EQ(label.type(), u8);

        const int value = static_cast<int>(label.scalar<unsigned char>());

        EXPECT_GE(value, 0);
        EXPECT_LE(value, 9);
    }
}

TEST_F(MNISTTest, FirstSampleLabel) {
    // The first image in the standard MNIST training set is a 5.
    auto sample = dataset->get(0);

    EXPECT_EQ(sample.target.data().scalar<unsigned char>(), 5);
}

// ==================== Bounds Checking ====================

TEST_F(MNISTTest, InvalidIndex) {
    EXPECT_THROW(dataset->get(60000), std::out_of_range);
}

TEST_F(MNISTTest, LargeInvalidIndex) {
    EXPECT_THROW(dataset->get(100000), std::out_of_range);
}

// ==================== Invalid Files ====================

TEST(MNISTDatasetTest, MissingImageFile) {
    EXPECT_THROW(MNISTDataset("nonexistent-images.idx3-ubyte",
                              (MNIST_DIR / "train-labels.idx1-ubyte").string()),
                 std::runtime_error);
}

TEST(MNISTDatasetTest, MissingLabelFile) {
    EXPECT_THROW(MNISTDataset((MNIST_DIR / "train-images.idx3-ubyte").string(),
                              "nonexistent-labels.idx1-ubyte"),
                 std::runtime_error);
}

// ==================== DataLoader Integration ====================

TEST_F(MNISTTest, BatchShape) {
    DataLoader loader(*dataset, 64, false);

    auto [images, labels] = *loader.begin();

    EXPECT_EQ(images.shape(), af::dim4(64, 784));
    EXPECT_EQ(labels.shape(), af::dim4(64, 1));
}

TEST_F(MNISTTest, BatchContents) {
    DataLoader loader(*dataset, 4, false);

    auto [images, labels] = *loader.begin();

    EXPECT_EQ(images.data().elements(), 4 * 784);
    EXPECT_EQ(labels.data().elements(), 4);

    // Compare the first sample in the batch to the original.
    auto original = dataset->get(0);

    af::array firstImage = images.data()(0, af::span);
    firstImage = af::flat(firstImage);

    af::array difference = af::abs(firstImage - original.input.data());

    EXPECT_LT(af::max<float>(difference), 1e-6f);
}

TEST_F(MNISTTest, IncompleteFinalBatch) {
    DataLoader loader(*dataset, 128, false);

    std::size_t total = 0;
    std::size_t lastBatchSize = 0;

    for (auto [images, labels] : loader) {
        const auto batchSize = static_cast<std::size_t>(images.shape()[0]);

        EXPECT_EQ(images.shape()[1], 784);
        EXPECT_EQ(labels.shape()[0], images.shape()[0]);

        total += batchSize;
        lastBatchSize = batchSize;
    }

    EXPECT_EQ(total, 60000);
    EXPECT_EQ(lastBatchSize, 96);
}
