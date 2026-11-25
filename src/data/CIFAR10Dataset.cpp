
#include "neuro/data/CIFAR10Dataset.hpp"

#include "neuro/Tensor.hpp"

#include <af/array.h>
#include <af/data.h>
#include <af/defines.h>
#include <arrayfire.h>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace neuro {

namespace {
constexpr std::size_t channels = 3;
constexpr std::size_t height = 32;
constexpr std::size_t width = 32;
constexpr std::size_t image_size = channels * height * width;
constexpr std::size_t record_size = 1 + image_size;

} // namespace

CIFAR10Dataset::CIFAR10Dataset(const std::vector<std::string>& image_files) {
    if (image_files.empty()) {
        throw std::invalid_argument("File list is empty");
    }

    for (const auto& filename : image_files) {
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file) {
            throw std::runtime_error("Failed to open file: " + filename);
        }
        std::streamsize size = file.tellg();
        if (size <= 0 || size % record_size != 0) {
            throw std::runtime_error("Invalid CIFAR-10 file size: " + filename);
        }
        file.seekg(0, std::ios::beg);
        std::vector<std::uint8_t> buffer(static_cast<std::size_t>(size));
        if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
            throw std::runtime_error("Failed to read file contents: " + filename);
        }
        const std::size_t num_records = buffer.size() / record_size;
        labels_.reserve(labels_.size() + num_records);
        images_.reserve(images_.size() + num_records * image_size);
        for (std::size_t record = 0; record < num_records; ++record) {
            const std::size_t offset = record * record_size;
            std::uint8_t label = buffer[offset];
            if (label > 9) {
                throw std::runtime_error("Invalid CIFAR-10 label");
            }
            labels_.push_back(label);
            images_.insert(images_.end(), buffer.begin() + offset + 1,
                           buffer.begin() + offset + record_size);
        }
    }
}

std::size_t CIFAR10Dataset::size() const {
    return labels_.size();
}

std::pair<Tensor, Tensor> CIFAR10Dataset::getBatch(std::span<const std::size_t> indices) const {
    if (indices.empty())
        throw std::invalid_argument("Batch cannot be empty");

    const std::size_t batch_size = indices.size();

    std::vector<std::uint8_t> image_buffer(batch_size * image_size);
    std::vector<std::uint8_t> label_buffer(batch_size);

    for (std::size_t b = 0; b < batch_size; ++b) {
        const std::size_t idx = indices[b];

        if (idx >= size())
            throw std::out_of_range("CIFAR-10 sample index out of range");

        for (std::size_t p = 0; p < image_size; ++p) {
            image_buffer[b + p * batch_size] = images_[idx * image_size + p];
        }

        label_buffer[b] = labels_[idx];
    }

    af::array batch_images(batch_size, image_size, image_buffer.data());
    af::array batch_labels(batch_size, label_buffer.data());

    batch_images = af::moddims(batch_images,
                               af::dim4(static_cast<dim_t>(batch_size), width, height, channels));

    batch_images = af::reorder(batch_images, 0, 3, 2, 1);

    batch_images = batch_images.as(f32) / 255.0f;

    return {Tensor(batch_images), Tensor(batch_labels)};
}

Dataset::Sample CIFAR10Dataset::get(std::size_t index) const {
    if (index >= size())
        throw std::out_of_range("CIFAR-10 sample index out of range");

    const std::uint8_t* pixels = images_.data() + index * image_size;

    af::array image(width, height, channels, pixels, afHost);
    image = af::reorder(image, 2, 1, 0);

    image = image.as(f32) / 255.0f;

    af::array label = af::constant(labels_[index], 1, u8);

    return {Tensor(image), Tensor(label)};
}

} // namespace neuro
