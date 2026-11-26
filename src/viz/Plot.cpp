#include "neuro/viz/Plot.hpp"

#include "neuro/viz/SvgDocument.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

constexpr std::array<std::string_view, 8> default_colors{
    "#2563eb", // Blue
    "#ea580c", // Orange
    "#16a34a", // Green
    "#dc2626", // Red
    "#9333ea", // Purple
    "#0891b2", // Cyan
    "#ca8a04", // Gold
    "#db2777"  // Pink
};

} // namespace

namespace neuro::viz {
Plot& Plot::addSeries(std::span<const double> x, std::span<const double> y,
                      const std::string& label, const std::string& color) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("X and y must have same length");
    }
    if (x.empty() || y.empty()) {
        throw std::invalid_argument("Series must contain at least one point");
    }
    std::string selected_color = color;
    if (color.empty())
        selected_color = std::string{default_colors[series_.size() % default_colors.size()]};
    Series series{{}, label, selected_color};
    series.points.reserve(x.size());
    for (std::size_t i = 0; i < x.size(); i++) {
        if (!std::isfinite(x[i]) || !std::isfinite(y[i])) {
            throw std::invalid_argument("Coordinates must be finite");
        }
        series.points.push_back({x[i], y[i]});
    }
    series_.push_back(std::move(series));
    return *this;
}

Plot::Bounds Plot::computeBounds() const {
    if (series_.empty()) {
        throw std::logic_error("No series available");
    }
    const auto& first = series_.front().points.front();
    Bounds bounds{first.x, first.x, first.y, first.y};
    for (const auto& series : series_) {
        for (const auto& point : series.points) {
            bounds.x_min = std::min(bounds.x_min, point.x);
            bounds.x_max = std::max(bounds.x_max, point.x);
            bounds.y_min = std::min(bounds.y_min, point.y);
            bounds.y_max = std::max(bounds.y_max, point.y);
        }
    }
    return bounds;
}

Point Plot::transformPoint(const Point& point, const Bounds& bounds, const PlotArea& area) {
    const double x_range = bounds.x_max - bounds.x_min;
    const double y_range = bounds.y_max - bounds.y_min;

    const double t_x = x_range == 0.0 ? 0.5 : (point.x - bounds.x_min) / x_range;
    const double t_y = y_range == 0.0 ? 0.5 : (point.y - bounds.y_min) / y_range;

    const double x_svg = area.left + t_x * area.width;
    const double y_svg = area.top + (1 - t_y) * area.height;
    return Point{x_svg, y_svg};
}

SvgDocument Plot::render(int width, int height) const {
    const double top_margin = title_.empty() ? 50 : 65;
    const double bottom_margin = xlabel_.empty() ? 50 : 75;
    const double left_margin = ylabel_.empty() ? 50 : 90;
    const double right_margin = 50;
    if (width <= (left_margin + right_margin) || height <= (top_margin + bottom_margin)) {
        throw std::invalid_argument("Dimensions too small to accommodate margins");
    }
    SvgDocument document(width, height);
    const Bounds bounds = padBounds(computeBounds());
    const PlotArea area{left_margin, top_margin, width - left_margin - right_margin,
                        height - top_margin - bottom_margin};
    document.rect(0, 0, width, height, "white");
    drawGridAndTicks(document, bounds, area);
    document.line(area.left, area.top, area.left, area.top + area.height, "black");
    document.line(area.left, area.top + area.height, area.left + area.width, area.top + area.height,
                  "black");
    for (const auto& series : series_) {
        std::vector<Point> transformed_points;
        transformed_points.reserve(series.points.size());
        for (const auto& point : series.points) {
            transformed_points.push_back(transformPoint(point, bounds, area));
        }
        if (transformed_points.size() == 1) {
            Point point = transformed_points[0];
            document.circle(point.x, point.y, 3.0, series.color);
        } else {
            document.polyline(transformed_points, series.color, 2.0);
        }
    }
    drawLabels(document, area);
    drawLegend(document, area);
    return document;
}

std::string Plot::str(int width, int height) const {
    return render(width, height).str();
}

void Plot::save(const std::string& filename, int width, int height) const {
    render(width, height).save(filename);
}

Plot::Bounds Plot::padBounds(const Bounds& bounds, double padding) {
    if (!std::isfinite(padding) || padding <= 0.0) {
        throw std::invalid_argument("Padding must be positive and finite");
    }

    const auto computePadding = [padding](double min, double max) {
        const double range = max - min;

        if (!std::isfinite(range)) {
            throw std::overflow_error("Cannot compute range");
        }

        return padding * (range == 0.0 ? std::max(std::abs(min), 1.0) : range);
    };

    const double x_padding = computePadding(bounds.x_min, bounds.x_max);
    const double y_padding = computePadding(bounds.y_min, bounds.y_max);

    const Bounds padded{bounds.x_min - x_padding, bounds.x_max + x_padding,
                        bounds.y_min - y_padding, bounds.y_max + y_padding};

    const auto validRange = [](double min, double max) {
        return std::isfinite(min) && std::isfinite(max) && std::isfinite(max - min) && min < max;
    };

    if (!validRange(padded.x_min, padded.x_max) || !validRange(padded.y_min, padded.y_max)) {
        throw std::overflow_error("Cannot compute padded bounds");
    }

    return padded;
}

double Plot::niceStep(double range, int target_intervals) {
    if (!std::isfinite(range) || range <= 0.0) {
        throw std::invalid_argument("Range must be positive and finite");
    }
    if (target_intervals <= 0) {
        throw std::invalid_argument("Target intervals must be positive");
    }

    const double raw_step = range / target_intervals;
    if (!std::isfinite(raw_step) || raw_step <= 0.0) {
        throw std::overflow_error("Cannot compute step");
    }
    const double exponent = std::floor(std::log10(raw_step));
    const double magnitude = std::pow(10.0, exponent);
    if (!std::isfinite(magnitude) || magnitude <= 0.0) {
        throw std::overflow_error("Cannot compute step");
    }
    const double fraction = raw_step / magnitude;

    constexpr double candidates[]{1.0, 2.0, 2.5, 5.0, 10.0};

    double step = 10.0 * magnitude;

    for (double candidate : candidates) {
        if (candidate >= fraction) {
            step = candidate * magnitude;
            break;
        }
    }

    if (!std::isfinite(step) || step <= 0.0) {
        throw std::overflow_error("Cannot compute step");
    }

    return step;
}

std::vector<double> Plot::generateTicks(double min, double max, int target_intervals) {
    if (!std::isfinite(min) || !std::isfinite(max) || min >= max) {
        throw std::invalid_argument("Invalid bounds");
    }

    const double step = niceStep(max - min, target_intervals);

    const double first = std::ceil(min / step);
    const double last = std::floor(max / step);

    constexpr double lower_bound = -0x1p63;
    constexpr double upper_bound = 0x1p63;

    const auto validIndex = [](double value) {
        return std::isfinite(value) && value >= lower_bound && value < upper_bound;
    };

    if (!validIndex(first) || !validIndex(last)) {
        throw std::overflow_error("Tick index out of range");
    }

    const auto first_index = static_cast<std::int64_t>(first);
    const auto last_index = static_cast<std::int64_t>(last);

    std::vector<double> ticks;

    for (auto i = first_index; i <= last_index; ++i) {
        const double tick = i * step;

        if (!std::isfinite(tick)) {
            throw std::overflow_error("Cannot compute tick");
        }

        ticks.push_back(tick);

        if (i == last_index) {
            break;
        }
    }

    return ticks;
}

void Plot::drawGridAndTicks(SvgDocument& document, const Bounds& bounds, const PlotArea& area) {
    const auto x_ticks = generateTicks(bounds.x_min, bounds.x_max);
    const auto y_ticks = generateTicks(bounds.y_min, bounds.y_max);
    constexpr double tick_length = 5;
    const double bottom = area.top + area.height;
    const double right = area.left + area.width;
    for (const double tick : x_ticks) {
        const double x = transformPoint({tick, 0}, bounds, area).x;
        document.line(x, area.top, x, bottom, "#e5e7eb");
        document.line(x, bottom, x, bottom + tick_length, "black");
        document.text(x, bottom + tick_length + 15, formatTick(tick), 12, "#333333",
                      TextAnchor::Middle);
    }
    for (const double tick : y_ticks) {
        const double y = transformPoint({0, tick}, bounds, area).y;
        document.line(area.left, y, right, y, "#e5e7eb");
        document.line(area.left - tick_length, y, area.left, y, "black");
        document.text(area.left - tick_length - 5, y + 4, formatTick(tick), 12, "#333333",
                      TextAnchor::End);
    }
}

std::string Plot::formatTick(double value) {
    if (value == 0.0 && std::signbit(value)) {
        value = 0;
    }
    return std::format("{:.6g}", value);
}

Plot& Plot::title(const std::string& title) {
    title_ = title;
    return *this;
}

Plot& Plot::xlabel(const std::string& label) {
    xlabel_ = label;
    return *this;
}

Plot& Plot::ylabel(const std::string& label) {
    ylabel_ = label;
    return *this;
}

void Plot::drawLabels(SvgDocument& document, const PlotArea& area) const {
    const std::string label_color = "#333333";
    if (!title_.empty())
        document.text(area.left + area.width / 2, area.top - 25, title_, 20, label_color,
                      TextAnchor::Middle);
    if (!xlabel_.empty())
        document.text(area.left + area.width / 2, area.top + area.height + 45, xlabel_, 14,
                      label_color, TextAnchor::Middle);
    if (!ylabel_.empty())
        document.text(area.left - 65, area.top + area.height / 2, ylabel_, 14, label_color,
                      TextAnchor::Middle, -90);
}

void Plot::drawLegend(SvgDocument& document, const PlotArea& area) const {
    constexpr double font_size = 12;
    constexpr double padding = 10;
    constexpr double row_height = 22;
    constexpr double gap = 8;
    constexpr double margin = 10;
    constexpr double sample_length = 20;

    std::size_t num_labels = 0;
    double max_label_width = 0;

    for (const auto& series : series_) {
        if (!series.label.empty()) {
            ++num_labels;
            const double label_width = series.label.size() * font_size * 0.6;
            max_label_width = std::max(max_label_width, label_width);
        }
    }

    if (num_labels == 0) {
        return;
    }

    const double legend_width = 2 * padding + sample_length + gap + max_label_width;

    const double legend_height = 2 * padding + num_labels * row_height;

    const double legend_x = area.left + area.width - margin - legend_width;

    const double legend_y = area.top + margin;

    document.rect(legend_x, legend_y, legend_width, legend_height, "white");

    const double sample_x = legend_x + padding;
    const double text_x = sample_x + sample_length + gap;

    double row_y = legend_y + padding + row_height / 2;

    for (const auto& series : series_) {
        if (series.label.empty()) {
            continue;
        }

        document.line(sample_x, row_y, sample_x + sample_length, row_y, series.color, 2.0);

        document.text(text_x, row_y + font_size / 3, series.label, font_size, "#333333");

        row_y += row_height;
    }
}
} // namespace neuro::viz