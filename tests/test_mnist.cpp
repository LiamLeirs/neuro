
#include "neuro/data/DataLoader.hpp"
#include "neuro/data/MNISTDataset.hpp"

#include <arrayfire.h>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <vector>

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

// ==================== Optimized Batch Loading ====================

TEST_F(MNISTTest, ShuffledBatchIndices) {
    const std::vector<std::size_t> indices = {42, 7, 100};

    auto [images, labels] = dataset->getBatch(indices);

    ASSERT_EQ(images.shape(), af::dim4(3, 784));
    ASSERT_EQ(labels.shape(), af::dim4(3));

    EXPECT_EQ(images.data().type(), f32);
    EXPECT_EQ(labels.data().type(), u8);

    for (std::size_t i = 0; i < indices.size(); ++i) {
        auto sample = dataset->get(indices[i]);

        af::array batch_image = af::flat(images.data()(i, af::span));
        af::array difference = af::abs(batch_image - sample.input.data());

        EXPECT_LT(af::max<float>(difference), 1e-6f);

        af::array batch_label = labels.data()(i);

        EXPECT_EQ(batch_label.scalar<std::uint8_t>(), sample.target.data().scalar<std::uint8_t>());
    }
}

TEST_F(MNISTTest, DuplicateBatchIndices) {
    const std::vector<std::size_t> indices = {5, 5, 10};

    auto [images, labels] = dataset->getBatch(indices);

    ASSERT_EQ(images.shape(), af::dim4(3, 784));
    ASSERT_EQ(labels.shape(), af::dim4(3));

    for (std::size_t i = 0; i < indices.size(); ++i) {
        auto sample = dataset->get(indices[i]);

        af::array batch_image = af::flat(images.data()(i, af::span));
        af::array difference = af::abs(batch_image - sample.input.data());

        EXPECT_LT(af::max<float>(difference), 1e-6f);

        af::array batch_label = labels.data()(i);

        EXPECT_EQ(batch_label.scalar<std::uint8_t>(), sample.target.data().scalar<std::uint8_t>());
    }
}

TEST_F(MNISTTest, EmptyBatchIndices) {
    const std::vector<std::size_t> indices;

    EXPECT_THROW(dataset->getBatch(indices), std::invalid_argument);
}

TEST_F(MNISTTest, InvalidBatchIndex) {
    const std::vector<std::size_t> indices = {0, 10, dataset->size()};

    EXPECT_THROW(dataset->getBatch(indices), std::out_of_range);
}

// ==================== Image Layout Fixture ====================

namespace {

class MNISTImageTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        dataset = std::make_unique<MNISTDataset>((MNIST_DIR / "train-images.idx3-ubyte").string(),
                                                 (MNIST_DIR / "train-labels.idx1-ubyte").string(),
                                                 MNISTLayout::Image);

        flattened = std::make_unique<MNISTDataset>(loadTrainingData());
    }

    static void TearDownTestSuite() {
        dataset.reset();
        flattened.reset();
    }

    static std::unique_ptr<MNISTDataset> dataset;
    static std::unique_ptr<MNISTDataset> flattened;

    void expectBatchMatchesSamples(const std::vector<std::size_t>& indices) {

        auto [images, labels] = dataset->getBatch(indices);

        const auto batch_size = static_cast<dim_t>(indices.size());

        ASSERT_EQ(images.shape(), af::dim4(batch_size, 1, 28, 28));

        ASSERT_EQ(labels.shape(), af::dim4(batch_size));

        EXPECT_EQ(images.data().type(), f32);
        EXPECT_EQ(labels.data().type(), u8);

        for (std::size_t i = 0; i < indices.size(); ++i) {
            auto sample = dataset->get(indices[i]);

            af::array batch_image =
                images.data()(static_cast<dim_t>(i), af::span, af::span, af::span);

            batch_image = af::moddims(batch_image, af::dim4(1, 28, 28));

            af::array difference = af::abs(batch_image - sample.input.data());

            EXPECT_LT(af::max<float>(difference), 1e-6f) << "Mismatch at batch index " << i;

            af::array batch_label = labels.data()(static_cast<dim_t>(i));

            EXPECT_EQ(batch_label.scalar<std::uint8_t>(),
                      sample.target.data().scalar<std::uint8_t>());
        }
    }
};

std::unique_ptr<MNISTDataset> MNISTImageTest::dataset;
std::unique_ptr<MNISTDataset> MNISTImageTest::flattened;

} // namespace

// ==================== Image Layout Loading ====================

TEST_F(MNISTImageTest, DatasetSize) {
    EXPECT_EQ(dataset->size(), 60000);
    EXPECT_EQ(dataset->size(), flattened->size());
}

TEST(MNISTDatasetTest, ImageLayoutTestDatasetSize) {
    MNISTDataset dataset((MNIST_DIR / "t10k-images.idx3-ubyte").string(),
                         (MNIST_DIR / "t10k-labels.idx1-ubyte").string(), MNISTLayout::Image);

    EXPECT_EQ(dataset.size(), 10000);
    EXPECT_EQ(dataset.get(0).input.shape(), af::dim4(1, 28, 28));
}

TEST(MNISTDatasetTest, RejectsInvalidLayout) {
    EXPECT_THROW(MNISTDataset((MNIST_DIR / "train-images.idx3-ubyte").string(),
                              (MNIST_DIR / "train-labels.idx1-ubyte").string(),
                              static_cast<MNISTLayout>(99)),
                 std::invalid_argument);
}

TEST(MNISTDatasetTest, DefaultLayoutRemainsFlattened) {
    auto dataset = loadTrainingData();

    EXPECT_EQ(dataset.get(0).input.shape(), af::dim4(784));
}

TEST(MNISTDatasetTest, ExplicitFlattenedLayoutMatchesDefault) {
    auto default_dataset = loadTrainingData();

    MNISTDataset explicit_dataset((MNIST_DIR / "train-images.idx3-ubyte").string(),
                                  (MNIST_DIR / "train-labels.idx1-ubyte").string(),
                                  MNISTLayout::Flattened);

    auto a = default_dataset.get(42);
    auto b = explicit_dataset.get(42);

    ASSERT_EQ(a.input.shape(), b.input.shape());

    EXPECT_LT(af::max<float>(af::abs(a.input.data() - b.input.data())), 1e-6f);

    EXPECT_EQ(a.target.data().scalar<std::uint8_t>(), b.target.data().scalar<std::uint8_t>());
}

// ==================== Individual Image Samples ====================

TEST_F(MNISTImageTest, InputShape) {
    auto sample = dataset->get(0);

    EXPECT_EQ(sample.input.shape(), af::dim4(1, 28, 28));

    EXPECT_EQ(sample.input.data().elements(), 784);
}

TEST_F(MNISTImageTest, TargetShapeAndType) {
    auto sample = dataset->get(0);

    EXPECT_EQ(sample.target.shape(), af::dim4(1));
    EXPECT_EQ(sample.target.data().type(), u8);
}

TEST_F(MNISTImageTest, PixelsAreNormalized) {
    for (std::size_t i : {0u, 100u, 1000u, 30000u, 59999u}) {
        auto sample = dataset->get(i);

        EXPECT_EQ(sample.input.data().type(), f32);
        EXPECT_GE(af::min<float>(sample.input.data()), 0.0f);
        EXPECT_LE(af::max<float>(sample.input.data()), 1.0f);
    }
}

TEST_F(MNISTImageTest, LabelsMatchFlattenedLayout) {
    for (std::size_t i : {0u, 7u, 42u, 1000u, 59999u}) {
        auto image_sample = dataset->get(i);
        auto flat_sample = flattened->get(i);

        EXPECT_EQ(image_sample.target.data().scalar<std::uint8_t>(),
                  flat_sample.target.data().scalar<std::uint8_t>());
    }
}

TEST_F(MNISTImageTest, SamplesDoNotRequireGrad) {
    auto sample = dataset->get(0);

    EXPECT_FALSE(sample.input.requires_grad());
    EXPECT_FALSE(sample.target.requires_grad());
}

// ==================== Pixel Orientation ====================

TEST_F(MNISTImageTest, CorrectPixelOrientation) {
    constexpr std::size_t height = 28;
    constexpr std::size_t width = 28;

    for (std::size_t index : {0u, 7u, 42u, 1000u, 59999u}) {
        auto image_sample = dataset->get(index);
        auto flat_sample = flattened->get(index);

        std::vector<float> image_pixels(height * width);
        std::vector<float> flat_pixels(height * width);

        image_sample.input.data().host(image_pixels.data());
        flat_sample.input.data().host(flat_pixels.data());

        for (std::size_t h = 0; h < height; ++h) {
            for (std::size_t w = 0; w < width; ++w) {
                // MNIST flattened representation: row-major.
                const auto flat_offset = h * width + w;

                // ArrayFire CHW representation: column-major.
                const auto image_offset = h + height * w;

                EXPECT_NEAR(image_pixels[image_offset], flat_pixels[flat_offset], 1e-6f)
                    << "Sample " << index << ", row " << h << ", column " << w;
            }
        }
    }
}

// ==================== Optimized Image Batching ====================

TEST_F(MNISTImageTest, BatchShape) {
    const std::vector<std::size_t> indices = {0, 1, 2, 3};

    auto [images, labels] = dataset->getBatch(indices);

    EXPECT_EQ(images.shape(), af::dim4(4, 1, 28, 28));
    EXPECT_EQ(labels.shape(), af::dim4(4));

    EXPECT_EQ(images.data().elements(), 4 * 784);
    EXPECT_EQ(labels.data().elements(), 4);
}

TEST_F(MNISTImageTest, SequentialBatchMatchesSamples) {
    expectBatchMatchesSamples({0, 1, 2, 3});
}

TEST_F(MNISTImageTest, ShuffledBatchMatchesSamples) {
    expectBatchMatchesSamples({42, 7, 100, 2});
}

TEST_F(MNISTImageTest, DuplicateBatchMatchesSamples) {
    expectBatchMatchesSamples({5, 5, 10, 5});
}

TEST_F(MNISTImageTest, SingleSampleBatch) {
    expectBatchMatchesSamples({42});
}

TEST_F(MNISTImageTest, LastSampleBatch) {
    expectBatchMatchesSamples({dataset->size() - 1});
}

TEST_F(MNISTImageTest, BatchMatchesStackedSamples) {
    const std::vector<std::size_t> indices = {42, 7, 100, 5};

    std::vector<Tensor> inputs;
    std::vector<Tensor> targets;

    for (auto index : indices) {
        auto sample = dataset->get(index);

        inputs.push_back(sample.input);
        targets.push_back(sample.target);
    }

    auto expected_images = Tensor::stack(inputs, 0);
    auto expected_labels = Tensor::stack(targets, 0);

    auto [images, labels] = dataset->getBatch(indices);

    ASSERT_EQ(images.shape(), expected_images.shape());
    ASSERT_EQ(labels.shape(), expected_labels.shape());

    EXPECT_LT(af::max<float>(af::abs(images.data() - expected_images.data())), 1e-6f);

    EXPECT_TRUE(af::allTrue<bool>(labels.data() == expected_labels.data()));
}

TEST_F(MNISTImageTest, BatchPixelOrientation) {
    const std::vector<std::size_t> indices = {42, 7, 100};

    auto [images, labels] = dataset->getBatch(indices);
    auto [flat_images, flat_labels] = flattened->getBatch(indices);

    constexpr std::size_t height = 28;
    constexpr std::size_t width = 28;

    const std::size_t batch_size = indices.size();

    std::vector<float> image_pixels(batch_size * height * width);
    std::vector<float> flat_pixels(batch_size * height * width);

    images.data().host(image_pixels.data());
    flat_images.data().host(flat_pixels.data());

    for (std::size_t b = 0; b < batch_size; ++b) {
        for (std::size_t h = 0; h < height; ++h) {
            for (std::size_t w = 0; w < width; ++w) {
                const auto flat_offset = b + batch_size * (h * width + w);

                const auto image_offset = b + batch_size * (h + height * w);

                EXPECT_NEAR(image_pixels[image_offset], flat_pixels[flat_offset], 1e-6f)
                    << "Batch index " << b << ", row " << h << ", column " << w;
            }
        }
    }

    EXPECT_TRUE(af::allTrue<bool>(labels.data() == flat_labels.data()));
}

TEST_F(MNISTImageTest, BatchPixelsAreNormalized) {
    const std::vector<std::size_t> indices = {0, 42, 100, 59999};

    auto [images, labels] = dataset->getBatch(indices);

    EXPECT_EQ(images.data().type(), f32);
    EXPECT_EQ(labels.data().type(), u8);

    EXPECT_GE(af::min<float>(images.data()), 0.0f);
    EXPECT_LE(af::max<float>(images.data()), 1.0f);
}

TEST_F(MNISTImageTest, BatchDoesNotRequireGrad) {
    const std::vector<std::size_t> indices = {0, 1, 2};

    auto [images, labels] = dataset->getBatch(indices);

    EXPECT_FALSE(images.requires_grad());
    EXPECT_FALSE(labels.requires_grad());
}

// ==================== Bounds Checking ====================

TEST_F(MNISTImageTest, InvalidIndex) {
    EXPECT_THROW(dataset->get(dataset->size()), std::out_of_range);
}

TEST_F(MNISTImageTest, EmptyBatchIndices) {
    const std::vector<std::size_t> indices;

    EXPECT_THROW(dataset->getBatch(indices), std::invalid_argument);
}

TEST_F(MNISTImageTest, InvalidBatchIndex) {
    const std::vector<std::size_t> indices = {0, 10, dataset->size()};

    EXPECT_THROW(dataset->getBatch(indices), std::out_of_range);
}

// ==================== DataLoader Integration ====================

TEST_F(MNISTImageTest, DataLoaderBatchShape) {
    DataLoader loader(*dataset, 64, false);

    auto [images, labels] = *loader.begin();

    EXPECT_EQ(images.shape(), af::dim4(64, 1, 28, 28));
    EXPECT_EQ(labels.shape(), af::dim4(64));
}

TEST_F(MNISTImageTest, DataLoaderBatchContents) {
    DataLoader loader(*dataset, 4, false);

    auto [images, labels] = *loader.begin();

    const std::vector<std::size_t> indices = {0, 1, 2, 3};

    auto [expected_images, expected_labels] = dataset->getBatch(indices);

    ASSERT_EQ(images.shape(), expected_images.shape());
    ASSERT_EQ(labels.shape(), expected_labels.shape());

    EXPECT_LT(af::max<float>(af::abs(images.data() - expected_images.data())), 1e-6f);

    EXPECT_TRUE(af::allTrue<bool>(labels.data() == expected_labels.data()));
}

TEST_F(MNISTImageTest, DataLoaderIncompleteFinalBatch) {
    DataLoader loader(*dataset, 1024, false);

    std::size_t total = 0;
    std::size_t last_batch_size = 0;

    for (auto [images, labels] : loader) {
        const auto batch_size = images.shape()[0];

        EXPECT_EQ(images.shape(), af::dim4(batch_size, 1, 28, 28));

        EXPECT_EQ(labels.shape(), af::dim4(batch_size));

        total += static_cast<std::size_t>(batch_size);
        last_batch_size = static_cast<std::size_t>(batch_size);
    }

    EXPECT_EQ(total, 60000);
    EXPECT_EQ(last_batch_size, 608);
}
