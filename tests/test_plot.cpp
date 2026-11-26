#include "neuro/Trainer.hpp"
#include "neuro/viz/Plot.hpp"

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using neuro::TrainingHistory;
using neuro::viz::Plot;

namespace {

const std::vector<double> x{1, 2, 3};
const std::vector<double> y{3, 2, 1};

bool contains(const std::string& svg, const std::string& value) {
    return svg.find(value) != std::string::npos;
}

} // namespace

TEST(PlotTest, RejectsEmptyPlot) {
    EXPECT_THROW(Plot{}.str(), std::logic_error);
}

TEST(PlotTest, RejectsInvalidSeries) {
    Plot plot;
    const std::vector<double> empty;
    const std::vector<double> short_y{1, 2};

    EXPECT_THROW(plot.addSeries(empty, empty), std::invalid_argument);
    EXPECT_THROW(plot.addSeries(x, short_y), std::invalid_argument);

    for (double invalid :
         {std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
          std::numeric_limits<double>::quiet_NaN()}) {
        const std::vector<double> values{1, 2, invalid};

        EXPECT_THROW(plot.addSeries(values, y), std::invalid_argument);
        EXPECT_THROW(plot.addSeries(x, values), std::invalid_argument);
    }
}

TEST(PlotTest, RejectedSeriesDoesNotModifyPlot) {
    Plot plot;
    plot.addSeries(x, y);

    const auto before = plot.str();
    const std::vector<double> invalid{1, 2};

    EXPECT_THROW(plot.addSeries(x, invalid), std::invalid_argument);
    EXPECT_EQ(plot.str(), before);
}

TEST(PlotTest, CopiesSeriesData) {
    Plot plot;

    {
        std::vector<double> local_x{1, 2, 3};
        std::vector<double> local_y{3, 2, 1};

        plot.addSeries(local_x, local_y);
    }

    Plot reference;
    reference.addSeries(x, y);

    EXPECT_EQ(plot.str(), reference.str());
}

TEST(PlotTest, RendersLineSeries) {
    Plot plot;
    plot.addSeries(x, y);

    const auto svg = plot.str();

    EXPECT_TRUE(contains(svg, "<polyline"));
    EXPECT_TRUE(contains(svg, "stroke=\"#2563eb\""));
    EXPECT_FALSE(contains(svg, "<circle"));
}

TEST(PlotTest, RendersSinglePoint) {
    Plot plot;
    const std::vector<double> point_x{2};
    const std::vector<double> point_y{5};

    plot.addSeries(point_x, point_y);

    const auto svg = plot.str();

    EXPECT_TRUE(contains(svg, "<circle"));
    EXPECT_TRUE(contains(svg, "fill=\"#2563eb\""));
    EXPECT_FALSE(contains(svg, "<polyline"));
}

TEST(PlotTest, SupportsConstantCoordinates) {
    const std::vector<double> constant{5, 5, 5};

    for (const auto& values : {constant, x}) {
        Plot plot;
        plot.addSeries(values, constant);

        EXPECT_NO_THROW(plot.str());
    }

    Plot plot;
    plot.addSeries(constant, y);
    EXPECT_NO_THROW(plot.str());
}

TEST(PlotTest, AssignsAutomaticColors) {
    Plot plot;

    plot.addSeries(x, y).addSeries(x, y).addSeries(x, y);

    const auto svg = plot.str();

    EXPECT_TRUE(contains(svg, "stroke=\"#2563eb\""));
    EXPECT_TRUE(contains(svg, "stroke=\"#ea580c\""));
    EXPECT_TRUE(contains(svg, "stroke=\"#16a34a\""));
}

TEST(PlotTest, CyclesColorPalette) {
    Plot plot;

    for (int i = 0; i < 9; ++i) {
        plot.addSeries(x, y);
    }

    const auto svg = plot.str();
    const std::string blue = "stroke=\"#2563eb\"";

    const auto first = svg.find(blue);
    ASSERT_NE(first, std::string::npos);

    EXPECT_NE(svg.find(blue, first + blue.size()), std::string::npos);
}

TEST(PlotTest, SupportsCustomColor) {
    Plot plot;
    plot.addSeries(x, y, "", "#123456");

    const auto svg = plot.str();

    EXPECT_TRUE(contains(svg, "stroke=\"#123456\""));
    EXPECT_FALSE(contains(svg, "stroke=\"#2563eb\""));
}

TEST(PlotTest, RendersTitlesAndAxisLabels) {
    Plot plot;

    plot.addSeries(x, y).title("Model Training").xlabel("Epoch").ylabel("Loss");

    const auto svg = plot.str();

    EXPECT_TRUE(contains(svg, ">Model Training</text>"));
    EXPECT_TRUE(contains(svg, ">Epoch</text>"));
    EXPECT_TRUE(contains(svg, ">Loss</text>"));
    EXPECT_TRUE(contains(svg, "transform=\"rotate(-90 "));
}

TEST(PlotTest, ReplacesAndClearsLabels) {
    Plot plot;

    plot.addSeries(x, y)
        .title("Original Title")
        .title("Updated Title")
        .xlabel("Epoch")
        .xlabel("")
        .ylabel("Loss")
        .ylabel("");

    const auto svg = plot.str();

    EXPECT_TRUE(contains(svg, ">Updated Title</text>"));
    EXPECT_FALSE(contains(svg, "Original Title"));
    EXPECT_FALSE(contains(svg, ">Epoch</text>"));
    EXPECT_FALSE(contains(svg, ">Loss</text>"));
    EXPECT_FALSE(contains(svg, "transform="));
}

TEST(PlotTest, RendersLegendForLabeledSeries) {
    Plot plot;

    plot.addSeries(x, y, "Training Loss").addSeries(x, y, "Validation Loss").addSeries(x, y);

    const auto svg = plot.str();

    EXPECT_TRUE(contains(svg, ">Training Loss</text>"));
    EXPECT_TRUE(contains(svg, ">Validation Loss</text>"));
}

TEST(PlotTest, OmitsLegendForUnlabeledSeries) {
    Plot plot;
    plot.addSeries(x, y);

    const auto svg = plot.str();

    const auto first = svg.find("<rect");
    ASSERT_NE(first, std::string::npos);

    // Only the canvas background should be present.
    EXPECT_EQ(svg.find("<rect", first + 1), std::string::npos);
}

TEST(PlotTest, EscapesLabels) {
    Plot plot;

    plot.addSeries(x, y, "Train & Test").title("Loss < 1");

    const auto svg = plot.str();

    EXPECT_TRUE(contains(svg, "Train &amp; Test"));
    EXPECT_TRUE(contains(svg, "Loss &lt; 1"));
}

TEST(PlotTest, SupportsCustomDimensions) {
    Plot plot;
    plot.addSeries(x, y);

    const auto svg = plot.str(1024, 768);

    EXPECT_TRUE(contains(svg, "viewBox=\"0 0 1024 768\""));
}

TEST(PlotTest, RejectsInsufficientDimensions) {
    Plot plot;
    plot.addSeries(x, y);

    EXPECT_THROW(plot.str(100, 600), std::invalid_argument);
    EXPECT_THROW(plot.str(800, 100), std::invalid_argument);

    plot.ylabel("Loss");

    EXPECT_THROW(plot.str(140, 600), std::invalid_argument);
}

TEST(PlotTest, RejectsOverflowingBounds) {
    const double limit = std::numeric_limits<double>::max();
    const std::vector<double> extreme{-limit, 0, limit};

    Plot plot;
    plot.addSeries(extreme, y);

    EXPECT_THROW(plot.str(), std::overflow_error);
}

TEST(PlotTest, RejectsUnrepresentableTickStep) {
    const double tiny = std::numeric_limits<double>::denorm_min();
    const std::vector<double> values{0, tiny};

    Plot plot;
    plot.addSeries(values, values);

    EXPECT_THROW(plot.str(), std::overflow_error);
}

TEST(PlotTest, RepeatedRenderingIsDeterministic) {
    Plot plot;

    plot.addSeries(x, y, "Loss").title("Training").xlabel("Epoch");

    EXPECT_EQ(plot.str(), plot.str());
}

TEST(PlotTest, SavesSvgDocument) {
    const auto path = std::filesystem::temp_directory_path() / "neuro_plot_test.svg";

    Plot plot;
    plot.addSeries(x, y, "Training Loss");

    plot.save(path.string());

    std::ifstream file(path);
    const std::string content{std::istreambuf_iterator<char>(file),
                              std::istreambuf_iterator<char>()};

    file.close();
    std::filesystem::remove(path);

    EXPECT_EQ(content, plot.str());
}

TEST(PlotTest, AddsSeriesFromTrainingHistory) {
    const TrainingHistory history{{{"loss", 1.5}, {"accuracy", 0.5}},
                                  {{"loss", 0.9}, {"accuracy", 0.7}},
                                  {{"loss", 0.4}, {"accuracy", 0.9}}};

    Plot plot;
    plot.addSeries(history, "loss");

    const std::vector<double> epochs{1, 2, 3};
    const std::vector<double> losses{1.5, 0.9, 0.4};

    Plot reference;
    reference.addSeries(epochs, losses, "loss");

    EXPECT_EQ(plot.str(), reference.str());
}

TEST(PlotTest, HistorySupportsCustomLabelAndColor) {
    const TrainingHistory history{{{"accuracy", 0.5}}, {{"accuracy", 0.8}}};

    Plot plot;
    plot.addSeries(history, "accuracy", "Training Accuracy", "#123456");

    const auto svg = plot.str();

    EXPECT_TRUE(contains(svg, ">Training Accuracy</text>"));
    EXPECT_TRUE(contains(svg, "stroke=\"#123456\""));
    EXPECT_FALSE(contains(svg, ">accuracy</text>"));
}

TEST(PlotTest, RejectsEmptyTrainingHistory) {
    Plot plot;
    const TrainingHistory history;

    EXPECT_THROW(plot.addSeries(history, "loss"), std::invalid_argument);
}

TEST(PlotTest, RejectsMissingHistoryMetric) {
    const TrainingHistory history{{{"loss", 1.5}}, {{"accuracy", 0.8}}, {{"loss", 0.4}}};

    Plot plot;
    plot.addSeries(x, y);

    const auto before = plot.str();

    EXPECT_THROW(plot.addSeries(history, "loss"), std::invalid_argument);

    EXPECT_EQ(plot.str(), before);
}

TEST(PlotTest, RejectsNonFiniteHistoryValues) {
    Plot plot;

    for (double invalid :
         {std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
          std::numeric_limits<double>::quiet_NaN()}) {
        const TrainingHistory history{{{"loss", 1.5}}, {{"loss", invalid}}};

        EXPECT_THROW(plot.addSeries(history, "loss"), std::invalid_argument);
    }
}

TEST(PlotTest, SupportsSingleEpochHistory) {
    const TrainingHistory history{{{"loss", 0.5}}};

    Plot plot;
    plot.addSeries(history, "loss");

    const auto svg = plot.str();

    EXPECT_TRUE(contains(svg, "<circle"));
    EXPECT_FALSE(contains(svg, "<polyline"));
}

TEST(PlotTest, CombinesHistoryAndNumericalSeries) {
    const TrainingHistory history{{{"loss", 3.0}}, {{"loss", 2.0}}, {{"loss", 1.0}}};

    Plot plot;
    plot.addSeries(history, "loss").addSeries(x, y, "Reference");

    Plot reference;
    reference.addSeries(x, y, "loss").addSeries(x, y, "Reference");

    EXPECT_EQ(plot.str(), reference.str());
}

TEST(PlotTest, ClearRemovesAllSeries) {
    Plot plot;

    plot.addSeries(x, y).addSeries(x, y);

    plot.clear();

    EXPECT_THROW(plot.str(), std::logic_error);
}

TEST(PlotTest, ClearRestoresDefaultState) {
    Plot plot;

    plot.addSeries(x, y, "Original Series")
        .addSeries(x, y, "Second Series")
        .title("Original Title")
        .xlabel("Original X")
        .ylabel("Original Y");

    plot.clear().addSeries(x, y);

    Plot reference;
    reference.addSeries(x, y);

    EXPECT_EQ(plot.str(), reference.str());
}

TEST(PlotTest, ClearEmptyPlotIsSafe) {
    Plot plot;

    EXPECT_NO_THROW(plot.clear());
    EXPECT_NO_THROW(plot.clear());

    plot.addSeries(x, y);

    Plot reference;
    reference.addSeries(x, y);

    EXPECT_EQ(plot.str(), reference.str());
}

#ifdef __linux__

TEST(PlotTest, ShowRejectsEmptyPlot) {
    Plot plot;
    EXPECT_THROW(plot.show(), std::logic_error);
}

TEST(PlotTest, ShowRejectsInvalidDimensions) {
    Plot plot;
    plot.addSeries(x, y);

    EXPECT_THROW(plot.show(0, 600), std::invalid_argument);
    EXPECT_THROW(plot.show(800, 0), std::invalid_argument);
}

#else

TEST(PlotTest, ShowRejectsUnsupportedPlatform) {
    Plot plot;
    plot.addSeries(x, y);

    EXPECT_THROW(plot.show(), std::runtime_error);
}

#endif