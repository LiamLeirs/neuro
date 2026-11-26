#include "neuro/nn/Module.hpp"
#include "neuro/nn/Sequential.hpp"

#include <gtest/gtest.h>
#include <memory>

using namespace neuro;
using namespace neuro::nn;

namespace {

class IdentityModule : public Module {
public:
    Tensor forward(const Tensor& input) const override {
        return input;
    }
};

} // namespace

TEST(ModuleModeTest, DefaultsToTraining) {
    IdentityModule module;
    EXPECT_TRUE(module.isTraining());
}

TEST(ModuleModeTest, SwitchesBetweenModes) {
    IdentityModule module;

    module.eval();
    EXPECT_FALSE(module.isTraining());

    module.train();
    EXPECT_TRUE(module.isTraining());

    module.train(false);
    EXPECT_FALSE(module.isTraining());

    module.train(true);
    EXPECT_TRUE(module.isTraining());
}

TEST(ModuleModeTest, IndependentModules) {
    IdentityModule first;
    IdentityModule second;

    first.eval();

    EXPECT_FALSE(first.isTraining());
    EXPECT_TRUE(second.isTraining());
}

TEST(ModuleModeTest, SequentialPropagatesMode) {
    Sequential model;

    auto first = std::make_unique<IdentityModule>();
    auto second = std::make_unique<IdentityModule>();

    auto* first_ptr = first.get();
    auto* second_ptr = second.get();

    model.add(std::move(first));
    model.add(std::move(second));

    model.eval();

    EXPECT_FALSE(model.isTraining());
    EXPECT_FALSE(first_ptr->isTraining());
    EXPECT_FALSE(second_ptr->isTraining());

    model.train();

    EXPECT_TRUE(model.isTraining());
    EXPECT_TRUE(first_ptr->isTraining());
    EXPECT_TRUE(second_ptr->isTraining());
}

TEST(ModuleModeTest, NestedSequentialPropagatesMode) {
    Sequential model;

    auto nested = std::make_unique<Sequential>();
    auto child = std::make_unique<IdentityModule>();

    auto* nested_ptr = nested.get();
    auto* child_ptr = child.get();

    nested->add(std::move(child));
    model.add(std::move(nested));

    model.eval();

    EXPECT_FALSE(model.isTraining());
    EXPECT_FALSE(nested_ptr->isTraining());
    EXPECT_FALSE(child_ptr->isTraining());

    model.train();

    EXPECT_TRUE(model.isTraining());
    EXPECT_TRUE(nested_ptr->isTraining());
    EXPECT_TRUE(child_ptr->isTraining());
}

TEST(ModuleModeTest, AddedModuleInheritsParentMode) {
    Sequential model;
    model.eval();

    auto child = std::make_unique<IdentityModule>();
    auto* child_ptr = child.get();

    model.add(std::move(child));

    EXPECT_FALSE(child_ptr->isTraining());
}

TEST(ModuleModeTest, EvalDispatchesThroughBaseReference) {
    Sequential model;

    auto child = std::make_unique<IdentityModule>();
    auto* child_ptr = child.get();

    model.add(std::move(child));

    Module& base = model;
    base.eval();

    EXPECT_FALSE(model.isTraining());
    EXPECT_FALSE(child_ptr->isTraining());
}