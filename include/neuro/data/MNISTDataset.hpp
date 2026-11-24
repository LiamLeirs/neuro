
#pragma once

#include "neuro/data/Dataset.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace neuro {

class MNISTDataset : public Dataset {
public:
    MNISTDataset(const std::string& image_filename, const std::string& label_filename);

    std::size_t size() const override;
    Sample get(std::size_t index) const override;

private:
    std::vector<std::uint8_t> images_;
    std::vector<std::uint8_t> labels_;
};

} // namespace neuro
