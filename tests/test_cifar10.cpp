
#include "neuro/data/CIFAR10Dataset.hpp"
#include "neuro/data/DataLoader.hpp"

#include <arrayfire.h>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace neuro;

namespace {

const std::filesystem::path CIFAR10_DIR =
    std::filesystem::path(__FILE__).parent_path().parent_path() / "data/cifar10";

constexpr std::size_t channels = 3;
constexpr std::size_t height = 32;
constexpr std::size_t width = 32;
constexpr std::size_t image_size = channels * height * width;
constexpr std::size_t record_size = 1 + image_size;
constexpr std::size_t samples_per_file = 10000;

std::vector<std::string> trainingFiles() {
    std::vector<std::string> files;

    for (int i = 1; i <= 5; ++i) {
        files.push_back((CIFAR10_DIR / ("data_batch_" + std::to_string(i) + ".bin")).string());
    }

    return files;
}

std::vector<std::string> testFiles() {
    return {(CIFAR10_DIR / "test_batch.bin").string()};
}

std::vector<std::uint8_t> readRawRecord(const std::string& filename, std::size_t index) {

    std::ifstream file(filename, std::ios::binary);

    if (!file)
        throw std::runtime_error("Cannot open CIFAR-10 file");

    file.seekg(static_cast<std::streamoff>(index * record_size), std::ios::beg);

    std::vector<std::uint8_t> record(record_size);

    if (!file.read(reinterpret_cast<char*>(record.data()),
                   static_cast<std::streamsize>(record.size()))) {
        throw std::runtime_error("Cannot read CIFAR-10 record");
    }

    return record;
}

std::vector<std::uint8_t> readTrainingRecord(std::size_t index) {

    const auto files = trainingFiles();

    return readRawRecord(files.at(index / samples_per_file), index % samples_per_file);
}

class CIFAR10Test : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        dataset = std::make_unique<CIFAR10Dataset>(trainingFiles());
    }

    static void TearDownTestSuite() {
        dataset.reset();
    }

    static std::unique_ptr<CIFAR10Dataset> dataset;

    void expectSampleMatchesRawRecord(std::size_t index) {
        auto sample = dataset->get(index);
        auto record = readTrainingRecord(index);

        ASSERT_EQ(sample.input.shape(), af::dim4(3, 32, 32));

        ASSERT_EQ(sample.input.data().type(), f32);
        ASSERT_EQ(sample.target.data().type(), u8);

        EXPECT_EQ(sample.target.data().scalar<std::uint8_t>(), record[0]);

        std::vector<float> pixels(image_size);
        sample.input.data().host(pixels.data());

        for (std::size_t c = 0; c < channels; ++c) {
            for (std::size_t h = 0; h < height; ++h) {
                for (std::size_t w = 0; w < width; ++w) {
                    // Original CIFAR-10 planar RGB format.
                    const std::size_t raw_offset = 1 + c * height * width + h * width + w;

                    // ArrayFire column-major CHW format.
                    const std::size_t tensor_offset = c + channels * (h + height * w);

                    const float expected = static_cast<float>(record[raw_offset]) / 255.0f;

                    EXPECT_NEAR(pixels[tensor_offset], expected, 1e-6f)
                        << "Sample " << index << ", channel " << c << ", row " << h << ", column "
                        << w;
                }
            }
        }
    }

    void expectBatchMatchesSamples(const std::vector<std::size_t>& indices) {

        auto [images, labels] = dataset->getBatch(indices);

        const auto batch_size = static_cast<dim_t>(indices.size());

        ASSERT_EQ(images.shape(), af::dim4(batch_size, 3, 32, 32));

        ASSERT_EQ(labels.shape(), af::dim4(batch_size));

        EXPECT_EQ(images.data().type(), f32);
        EXPECT_EQ(labels.data().type(), u8);

        for (std::size_t i = 0; i < indices.size(); ++i) {
            auto sample = dataset->get(indices[i]);

            af::array batch_image =
                images.data()(static_cast<dim_t>(i), af::span, af::span, af::span);

            batch_image = af::moddims(batch_image, af::dim4(3, 32, 32));

            af::array difference = af::abs(batch_image - sample.input.data());

            EXPECT_LT(af::max<float>(difference), 1e-6f) << "Batch index " << i;

            af::array batch_label = labels.data()(static_cast<dim_t>(i));

            EXPECT_EQ(batch_label.scalar<std::uint8_t>(),
                      sample.target.data().scalar<std::uint8_t>());
        }
    }
};

std::unique_ptr<CIFAR10Dataset> CIFAR10Test::dataset;

} // namespace

// ==================== Dataset Loading ====================

TEST_F(CIFAR10Test, TrainingDatasetSize) {
    EXPECT_EQ(dataset->size(), 50000);
}

TEST(CIFAR10DatasetTest, TestDatasetSize) {
    CIFAR10Dataset dataset(testFiles());

    EXPECT_EQ(dataset.size(), 10000);
}

TEST(CIFAR10DatasetTest, SingleFileLoading) {
    const auto files = trainingFiles();

    CIFAR10Dataset dataset({files[0]});

    EXPECT_EQ(dataset.size(), 10000);
}

TEST(CIFAR10DatasetTest, MultipleFileLoading) {
    const auto files = trainingFiles();

    CIFAR10Dataset dataset({files[0], files[1]});

    EXPECT_EQ(dataset.size(), 20000);
}

TEST(CIFAR10DatasetTest, FileOrderIsPreserved) {
    const auto files = trainingFiles();

    CIFAR10Dataset combined({files[1], files[0]});
    CIFAR10Dataset first({files[0]});
    CIFAR10Dataset second({files[1]});

    auto combined_first = combined.get(0);
    auto combined_second = combined.get(10000);

    auto expected_first = second.get(0);
    auto expected_second = first.get(0);

    EXPECT_LT(af::max<float>(af::abs(combined_first.input.data() - expected_first.input.data())),
              1e-6f);

    EXPECT_LT(af::max<float>(af::abs(combined_second.input.data() - expected_second.input.data())),
              1e-6f);

    EXPECT_EQ(combined_first.target.data().scalar<std::uint8_t>(),
              expected_first.target.data().scalar<std::uint8_t>());

    EXPECT_EQ(combined_second.target.data().scalar<std::uint8_t>(),
              expected_second.target.data().scalar<std::uint8_t>());
}

// ==================== Individual Samples ====================

TEST_F(CIFAR10Test, InputShape) {
    auto sample = dataset->get(0);

    EXPECT_EQ(sample.input.shape(), af::dim4(3, 32, 32));
    EXPECT_EQ(sample.input.data().elements(), 3072);
}

TEST_F(CIFAR10Test, TargetShapeAndType) {
    auto sample = dataset->get(0);

    EXPECT_EQ(sample.target.shape(), af::dim4(1));
    EXPECT_EQ(sample.target.data().type(), u8);
}

TEST_F(CIFAR10Test, PixelsAreNormalized) {
    for (std::size_t i : {0u, 100u, 10000u, 30000u, 49999u}) {
        auto sample = dataset->get(i);

        EXPECT_EQ(sample.input.data().type(), f32);
        EXPECT_GE(af::min<float>(sample.input.data()), 0.0f);
        EXPECT_LE(af::max<float>(sample.input.data()), 1.0f);
    }
}

TEST_F(CIFAR10Test, LabelsAreValid) {
    for (std::size_t i : {0u, 100u, 10000u, 30000u, 49999u}) {
        auto sample = dataset->get(i);

        ASSERT_EQ(sample.target.data().type(), u8);

        EXPECT_LE(static_cast<int>(sample.target.data().scalar<std::uint8_t>()), 9);
    }
}

TEST_F(CIFAR10Test, FirstSampleLabel) {
    // First sample in the official CIFAR-10 training set:
    // frog (class 6).
    auto sample = dataset->get(0);

    EXPECT_EQ(sample.target.data().scalar<std::uint8_t>(), 6);
}

TEST_F(CIFAR10Test, SamplesDoNotRequireGrad) {
    auto sample = dataset->get(0);

    EXPECT_FALSE(sample.input.requires_grad());
    EXPECT_FALSE(sample.target.requires_grad());
}

// ==================== Pixel Orientation ====================

TEST_F(CIFAR10Test, CorrectPixelOrientation) {
    for (std::size_t index : {0u, 7u, 42u, 1000u, 49999u}) {
        expectSampleMatchesRawRecord(index);
    }
}

TEST_F(CIFAR10Test, FileBoundarySamples) {
    for (std::size_t boundary = samples_per_file; boundary < dataset->size();
         boundary += samples_per_file) {

        expectSampleMatchesRawRecord(boundary - 1);
        expectSampleMatchesRawRecord(boundary);
    }
}

TEST_F(CIFAR10Test, CorrectChannelOrdering) {
    auto sample = dataset->get(42);
    auto record = readTrainingRecord(42);

    for (std::size_t c = 0; c < channels; ++c) {
        af::array channel = sample.input.data()(static_cast<dim_t>(c), af::span, af::span);

        const float actual_sum = af::sum<float>(channel);

        float expected_sum = 0.0f;

        for (std::size_t p = 0; p < height * width; ++p) {
            expected_sum += static_cast<float>(record[1 + c * height * width + p]) / 255.0f;
        }

        EXPECT_NEAR(actual_sum, expected_sum, 1e-2f) << "Channel " << c;
    }
}

// ==================== Optimized Batch Loading ====================

TEST_F(CIFAR10Test, BatchShape) {
    const std::vector<std::size_t> indices = {0, 1, 2, 3};

    auto [images, labels] = dataset->getBatch(indices);

    EXPECT_EQ(images.shape(), af::dim4(4, 3, 32, 32));
    EXPECT_EQ(labels.shape(), af::dim4(4));

    EXPECT_EQ(images.data().elements(), 4 * 3072);
    EXPECT_EQ(labels.data().elements(), 4);
}

TEST_F(CIFAR10Test, SequentialBatchMatchesSamples) {
    expectBatchMatchesSamples({0, 1, 2, 3});
}

TEST_F(CIFAR10Test, ShuffledBatchMatchesSamples) {
    expectBatchMatchesSamples({42, 7, 100, 2});
}

TEST_F(CIFAR10Test, DuplicateBatchMatchesSamples) {
    expectBatchMatchesSamples({5, 5, 10, 5});
}

TEST_F(CIFAR10Test, SingleSampleBatch) {
    expectBatchMatchesSamples({42});
}

TEST_F(CIFAR10Test, LastSampleBatch) {
    expectBatchMatchesSamples({dataset->size() - 1});
}

TEST_F(CIFAR10Test, BatchAcrossFileBoundaries) {
    expectBatchMatchesSamples({9999, 10000, 19999, 20000, 29999, 30000, 39999, 40000});
}

TEST_F(CIFAR10Test, BatchMatchesStackedSamples) {
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

TEST_F(CIFAR10Test, BatchPixelOrientation) {
    const std::vector<std::size_t> indices = {42, 10000, 7, 49999};

    auto [images, labels] = dataset->getBatch(indices);

    const std::size_t batch_size = indices.size();

    ASSERT_EQ(images.shape(), af::dim4(static_cast<dim_t>(batch_size), 3, 32, 32));

    ASSERT_EQ(images.data().type(), f32);
    ASSERT_EQ(labels.data().type(), u8);

    std::vector<float> pixels(batch_size * image_size);
    std::vector<std::uint8_t> batch_labels(batch_size);

    images.data().host(pixels.data());
    labels.data().host(batch_labels.data());

    for (std::size_t b = 0; b < batch_size; ++b) {
        auto record = readTrainingRecord(indices[b]);

        EXPECT_EQ(batch_labels[b], record[0]);

        for (std::size_t c = 0; c < channels; ++c) {
            for (std::size_t h = 0; h < height; ++h) {
                for (std::size_t w = 0; w < width; ++w) {
                    const std::size_t raw_offset = 1 + c * height * width + h * width + w;

                    // ArrayFire column-major NCHW.
                    const std::size_t tensor_offset =
                        b + batch_size * (c + channels * (h + height * w));

                    const float expected = static_cast<float>(record[raw_offset]) / 255.0f;

                    EXPECT_NEAR(pixels[tensor_offset], expected, 1e-6f)
                        << "Batch index " << b << ", channel " << c << ", row " << h << ", column "
                        << w;
                }
            }
        }
    }
}

TEST_F(CIFAR10Test, BatchPixelsAreNormalized) {
    const std::vector<std::size_t> indices = {0, 42, 100, 49999};

    auto [images, labels] = dataset->getBatch(indices);

    EXPECT_EQ(images.data().type(), f32);
    EXPECT_EQ(labels.data().type(), u8);

    EXPECT_GE(af::min<float>(images.data()), 0.0f);
    EXPECT_LE(af::max<float>(images.data()), 1.0f);
}

TEST_F(CIFAR10Test, BatchDoesNotRequireGrad) {
    const std::vector<std::size_t> indices = {0, 1, 2};

    auto [images, labels] = dataset->getBatch(indices);

    EXPECT_FALSE(images.requires_grad());
    EXPECT_FALSE(labels.requires_grad());
}

// ==================== Validation ====================

TEST(CIFAR10DatasetTest, EmptyFileList) {
    const std::vector<std::string> files;

    EXPECT_THROW(CIFAR10Dataset{files}, std::invalid_argument);
}

TEST(CIFAR10DatasetTest, MissingFile) {
    EXPECT_THROW(CIFAR10Dataset({"nonexistent-cifar10.bin"}), std::runtime_error);
}

TEST(CIFAR10DatasetTest, MissingFileAfterValidFile) {
    const auto files = trainingFiles();

    EXPECT_THROW(CIFAR10Dataset({files[0], "nonexistent-cifar10.bin"}), std::runtime_error);
}

TEST_F(CIFAR10Test, InvalidIndex) {
    EXPECT_THROW(dataset->get(dataset->size()), std::out_of_range);
}

TEST_F(CIFAR10Test, LargeInvalidIndex) {
    EXPECT_THROW(dataset->get(100000), std::out_of_range);
}

TEST_F(CIFAR10Test, EmptyBatchIndices) {
    const std::vector<std::size_t> indices;

    EXPECT_THROW(dataset->getBatch(indices), std::invalid_argument);
}

TEST_F(CIFAR10Test, InvalidBatchIndex) {
    const std::vector<std::size_t> indices = {0, 10, dataset->size()};

    EXPECT_THROW(dataset->getBatch(indices), std::out_of_range);
}

// ==================== DataLoader Integration ====================

TEST_F(CIFAR10Test, DataLoaderBatchShape) {
    DataLoader loader(*dataset, 64, false);

    auto [images, labels] = *loader.begin();

    EXPECT_EQ(images.shape(), af::dim4(64, 3, 32, 32));
    EXPECT_EQ(labels.shape(), af::dim4(64));
}

TEST_F(CIFAR10Test, DataLoaderBatchContents) {
    DataLoader loader(*dataset, 4, false);

    auto [images, labels] = *loader.begin();

    const std::vector<std::size_t> indices = {0, 1, 2, 3};

    auto [expected_images, expected_labels] = dataset->getBatch(indices);

    ASSERT_EQ(images.shape(), expected_images.shape());
    ASSERT_EQ(labels.shape(), expected_labels.shape());

    EXPECT_LT(af::max<float>(af::abs(images.data() - expected_images.data())), 1e-6f);

    EXPECT_TRUE(af::allTrue<bool>(labels.data() == expected_labels.data()));
}

TEST_F(CIFAR10Test, DataLoaderIncompleteFinalBatch) {
    DataLoader loader(*dataset, 1024, false);

    std::size_t total = 0;
    std::size_t last_batch_size = 0;

    for (auto [images, labels] : loader) {
        const auto batch_size = images.shape()[0];

        EXPECT_EQ(images.shape(), af::dim4(batch_size, 3, 32, 32));

        EXPECT_EQ(labels.shape(), af::dim4(batch_size));

        total += static_cast<std::size_t>(batch_size);
        last_batch_size = static_cast<std::size_t>(batch_size);
    }

    EXPECT_EQ(total, 50000);
    EXPECT_EQ(last_batch_size, 848);
}

TEST(CIFAR10DatasetTest, TestDatasetDataLoader) {
    CIFAR10Dataset dataset(testFiles());
    DataLoader loader(dataset, 128, false);

    auto [images, labels] = *loader.begin();

    EXPECT_EQ(images.shape(), af::dim4(128, 3, 32, 32));
    EXPECT_EQ(labels.shape(), af::dim4(128));
}
