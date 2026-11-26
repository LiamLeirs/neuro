
#pragma once

#include "neuro/Trainer.hpp"
#include "neuro/viz/SvgDocument.hpp"

#include <vector>
namespace neuro::viz {
class Plot {
public:
    Plot& addSeries(std::span<const double> x, std::span<const double> y,
                    const std::string& label = "", const std::string& color = "");

    Plot& addSeries(const TrainingHistory& history, const std::string& metric,
                    const std::string& label = "", const std::string& color = "");

    std::string str(int width = 800, int height = 600) const;

    void save(const std::string& filename, int width = 800, int height = 600) const;
    Plot& clear();
    Plot& title(const std::string& title);
    Plot& xlabel(const std::string& label);
    Plot& ylabel(const std::string& label);

private:
    struct Series {
        std::vector<Point> points;
        std::string label;
        std::string color;
    };
    struct Bounds {
        double x_min;
        double x_max;
        double y_min;
        double y_max;
    };
    struct PlotArea {
        double left;
        double top;
        double width;
        double height;
    };
    static Point transformPoint(const Point& point, const Bounds& bounds, const PlotArea& area);
    Bounds computeBounds() const;
    static Bounds padBounds(const Bounds& bounds, double padding = 0.05);
    static double niceStep(double range, int target_intervals = 5);
    static std::vector<double> generateTicks(double min, double max, int target_intervals = 5);
    static void drawGridAndTicks(SvgDocument& document, const Bounds& bounds, const PlotArea& area);
    SvgDocument render(int width, int height) const;
    static std::string formatTick(double value);
    void drawLabels(SvgDocument& document, const PlotArea& area) const;
    void drawLegend(SvgDocument& document, const PlotArea& area) const;
    std::vector<Series> series_;

    std::string title_;
    std::string xlabel_;
    std::string ylabel_;
};

} // namespace neuro::viz
