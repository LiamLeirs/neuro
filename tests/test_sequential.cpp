
#include "neuro/Tensor.hpp"
#include "neuro/nn/Linear.hpp"
#include "neuro/nn/Module.hpp"
#include "neuro/nn/ReLU.hpp"
#include "neuro/nn/Sequential.hpp"
#include "neuro/optim/SGD.hpp"

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace neuro;

namespace {

// Construct matrices without ambiguous initializer-list overloads.
// Input values are specified in row-major order.
Tensor matrix(int rows, int cols, const std::vector<float>& values, bool requires_grad = false) {
    if (values.size() != static_cast<std::size_t>(rows * cols)) {
        throw std::invalid_argument("Incorrect number of matrix elements");
    }

    af::array data(cols, rows, values.data(), afHost);
    return Tensor(data.T(), requires_grad);
}

void expectArrayNear(const af::array& actual, const af::array& expected, float tolerance = 1e-5f) {
    ASSERT_EQ(actual.dims(), expected.dims());
    EXPECT_TRUE(af::allTrue<bool>(af::abs(actual - expected) < tolerance));
}

// Simple modules for verifying execution order.
class AddOne : public nn::Module {
public:
    Tensor forward(const Tensor& input) const override {
        return input + 1.0;
    }
};

class MultiplyTwo : public nn::Module {
public:
    Tensor forward(const Tensor& input) const override {
        return input * 2.0;
    }
};

} // namespace

TEST(SequentialTest, EmptySequentialReturnsInput) {
    nn::Sequential model;
    Tensor input = matrix(2, 2, {1, 2, 3, 4});

    Tensor output = model(input);

    EXPECT_EQ(output.shape(), input.shape());
    expectArrayNear(output.data(), input.data());
    EXPECT_TRUE(model.parameters().empty());
}

TEST(SequentialTest, AppliesModulesInOrder) {
    nn::Sequential model{AddOne(), MultiplyTwo()};

    Tensor input = matrix(2, 2, {1, 2, 3, 4});
    Tensor output = model(input);

    // (input + 1) * 2
    Tensor expected = matrix(2, 2, {4, 6, 8, 10});

    expectArrayNear(output.data(), expected.data());
}

TEST(SequentialTest, CollectsParametersFromLinearLayers) {
    nn::Sequential model{nn::Linear(2, 4), nn::ReLU(), nn::Linear(4, 1)};

    auto params = model.parameters();

    ASSERT_EQ(params.size(), 4);

    EXPECT_EQ(params[0].shape(), af::dim4(2, 4));
    EXPECT_EQ(params[1].shape(), af::dim4(1, 4));
    EXPECT_EQ(params[2].shape(), af::dim4(4, 1));
    EXPECT_EQ(params[3].shape(), af::dim4(1, 1));

    for (const auto& param : params) {
        EXPECT_TRUE(param.requires_grad());
    }
}

TEST(SequentialTest, BackpropagatesThroughMultipleLayers) {
    nn::Sequential model{nn::Linear(2, 4), nn::ReLU(), nn::Linear(4, 1)};

    Tensor input = matrix(2, 2, {1, 2, 3, 4});

    Tensor loss = model(input).sum();
    loss.backward();

    auto params = model.parameters();
    ASSERT_EQ(params.size(), 4);

    for (const auto& param : params) {
        EXPECT_EQ(param.grad().dims(), param.data().dims());
        EXPECT_FALSE(af::anyTrue<bool>(af::isNaN(param.grad())));
        EXPECT_FALSE(af::anyTrue<bool>(af::isInf(param.grad())));
    }

    // The final layer's bias must receive a nonzero gradient.
    EXPECT_TRUE(af::anyTrue<bool>(af::abs(params[3].grad()) > 1e-7f));
}

TEST(SequentialTest, SGDUpdatesModelParameters) {
    nn::Sequential model{nn::Linear(2, 1)};
    SGD optimizer(model.parameters(), 0.01);

    Tensor input = matrix(1, 2, {1, 2});

    auto params = model.parameters();
    ASSERT_EQ(params.size(), 2);

    af::array before = params[1].data().copy();

    optimizer.zeroGrad();

    // Sum of the linear output has bias gradient exactly 1.
    Tensor loss = model(input).sum();

    loss.backward();

    EXPECT_NEAR(params[1].grad().scalar<float>(), 1.0f, 1e-5f);

    optimizer.step();

    af::array after = model.parameters()[1].data();

    EXPECT_NEAR((before - after).scalar<float>(), 0.01f, 1e-5f);
}

TEST(SequentialTest, RejectsNullModule) {
    nn::Sequential model;

    EXPECT_THROW(model.add(nullptr), std::invalid_argument);
}

TEST(SequentialTest, SupportsMoveConstruction) {
    nn::Sequential original{nn::Linear(2, 4), nn::ReLU(), nn::Linear(4, 1)};

    nn::Sequential moved(std::move(original));

    EXPECT_EQ(moved.parameters().size(), 4);

    Tensor input = matrix(1, 2, {1, 2});
    EXPECT_EQ(moved(input).shape(), af::dim4(1, 1));
}

TEST(SequentialTest, SupportsDynamicModuleAddition) {
    nn::Sequential model;

    model.add(std::make_unique<nn::Linear>(2, 4));
    model.add(std::make_unique<nn::ReLU>());
    model.add(std::make_unique<nn::Linear>(4, 1));

    EXPECT_EQ(model.parameters().size(), 4);

    Tensor input = matrix(1, 2, {1, 2});
    EXPECT_EQ(model(input).shape(), af::dim4(1, 1));
}
