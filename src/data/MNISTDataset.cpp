
#include "neuro/data/MNISTDataset.hpp"

#include "neuro/Tensor.hpp"

#include <af/array.h>
#include <af/data.h>
#include <af/defines.h>
#include <arrayfire.h>
#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace neuro {

namespace {

std::uint32_t read_uint32(std::ifstream& file) {
    std::uint8_t bytes[4];

    if (!file.read(reinterpret_cast<char*>(bytes), 4))
        throw std::runtime_error("Invalid MNIST header");

    return (std::uint32_t(bytes[0]) << 24) | (std::uint32_t(bytes[1]) << 16) |
           (std::uint32_t(bytes[2]) << 8) | std::uint32_t(bytes[3]);
}

} // namespace

MNISTDataset::MNISTDataset(const std::string& image_filename, const std::string& label_filename,
                           MNISTLayout layout)
    : layout_(layout) {

    if (layout_ != MNISTLayout::Flattened && layout_ != MNISTLayout::Image) {
        throw std::invalid_argument("Invalid layout supplied");
    }

    std::ifstream images(image_filename, std::ios::binary);
    std::ifstream labels(label_filename, std::ios::binary);

    if (!images || !labels)
        throw std::runtime_error("Cannot open MNIST files");

    if (read_uint32(images) != 2051 || read_uint32(labels) != 2049)
        throw std::runtime_error("Invalid MNIST magic number");

    const auto count = read_uint32(images);
    const auto label_count = read_uint32(labels);
    const auto rows = read_uint32(images);
    const auto cols = read_uint32(images);

    if (count != label_count || rows != 28 || cols != 28)
        throw std::runtime_error("Invalid MNIST dimensions or sample count");

    images_.resize(static_cast<std::size_t>(count) * rows * cols);
    labels_.resize(count);

    if (!images.read(reinterpret_cast<char*>(images_.data()), images_.size()) ||
        !labels.read(reinterpret_cast<char*>(labels_.data()), labels_.size()))
        throw std::runtime_error("Incomplete MNIST data");

    for (auto label : labels_) {
        if (label > 9)
            throw std::runtime_error("Invalid MNIST label");
    }
}

std::size_t MNISTDataset::size() const {
    return labels_.size();
}

std::pair<Tensor, Tensor> MNISTDataset::getBatch(std::span<const std::size_t> indices) const {
    if (indices.empty())
        throw std::invalid_argument("Batch cannot be empty");

    const std::size_t batch_size = indices.size();
    constexpr std::size_t image_size = 28 * 28;

    std::vector<std::uint8_t> image_buffer(batch_size * image_size);
    std::vector<std::uint8_t> label_buffer(batch_size);

    for (std::size_t b = 0; b < batch_size; ++b) {
        const std::size_t idx = indices[b];

        if (idx >= size())
            throw std::out_of_range("MNIST sample index out of range");

        for (std::size_t p = 0; p < image_size; ++p) {
            image_buffer[b + p * batch_size] = images_[idx * image_size + p];
        }

        label_buffer[b] = labels_[idx];
    }

    af::array batch_images(batch_size, image_size, image_buffer.data());
    af::array batch_labels(batch_size, label_buffer.data());

    if (layout_ == MNISTLayout::Image) {
        batch_images =
            af::moddims(batch_images, af::dim4(static_cast<dim_t>(batch_size), 1, 28, 28));

        batch_images = af::reorder(batch_images, 0, 1, 3, 2);
    }

    batch_images = batch_images.as(f32) / 255.0f;

    return {Tensor(batch_images), Tensor(batch_labels)};
}

Dataset::Sample MNISTDataset::get(std::size_t index) const {
    if (index >= size())
        throw std::out_of_range("MNIST sample index out of range");

    constexpr std::size_t image_size = 28 * 28;

    const std::uint8_t* pixels = images_.data() + index * image_size;
    af::array image;

    if (layout_ == MNISTLayout::Flattened) {
        image = af::array(image_size, pixels, afHost);
    } else if (layout_ == MNISTLayout::Image) {
        image = af::array(28, 28, pixels, afHost);
        image = af::moddims(image.T(), af::dim4(1, 28, 28));
    }

    image = image.as(f32) / 255.0f;

    af::array label = af::constant(labels_[index], 1, u8);

    return {Tensor(image), Tensor(label)};
}

} // namespace neuro
