
#pragma once

#include "neuro/Tensor.hpp"
#include "neuro/data/Dataset.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace neuro {

enum class MNISTLayout { Flattened, Image };

class MNISTDataset : public Dataset {
public:
    MNISTDataset(const std::string& image_filename, const std::string& label_filename,
                 MNISTLayout layout = MNISTLayout::Flattened);

    std::size_t size() const override;
    Sample get(std::size_t index) const override;
    std::pair<Tensor, Tensor> getBatch(std::span<const std::size_t> indices) const override;

private:
    std::vector<std::uint8_t> images_;
    std::vector<std::uint8_t> labels_;
    MNISTLayout layout_;
};

} // namespace neuro
