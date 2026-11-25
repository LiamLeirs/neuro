
#pragma once

#include "neuro/Tensor.hpp"
#include "neuro/data/Dataset.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace neuro {

class CIFAR10Dataset : public Dataset {
public:
    explicit CIFAR10Dataset(const std::vector<std::string>& image_files);
    std::size_t size() const override;
    Sample get(std::size_t index) const override;
    std::pair<Tensor, Tensor> getBatch(std::span<const std::size_t> indices) const override;

private:
    std::vector<std::uint8_t> images_;
    std::vector<std::uint8_t> labels_;
};

} // namespace neuro
