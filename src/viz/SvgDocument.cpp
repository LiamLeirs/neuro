
#include "neuro/viz/SvgDocument.hpp"

#include <cmath>
#include <format>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string_view>

namespace {

std::string escapeXml(const std::string& value) {
    std::string output;
    output.reserve(value.size());

    for (char c : value) {
        switch (c) {
        case '&':
            output += "&amp;";
            break;
        case '<':
            output += "&lt;";
            break;
        case '>':
            output += "&gt;";
            break;
        case '"':
            output += "&quot;";
            break;
        case '\'':
            output += "&apos;";
            break;
        default:
            output += c;
        }
    }

    return output;
}

std::string_view toSvgAnchor(neuro::viz::TextAnchor anchor) {
    using neuro::viz::TextAnchor;

    switch (anchor) {
    case TextAnchor::Start:
        return "start";
    case TextAnchor::Middle:
        return "middle";
    case TextAnchor::End:
        return "end";
    default:
        throw std::invalid_argument("Invalid text anchor");
    }
}

} // namespace

namespace neuro::viz {

SvgDocument::SvgDocument(int width, int height) : width_(width), height_(height) {
    if (width_ <= 0 || height_ <= 0) {
        throw std::invalid_argument("Width and height must be greater than zero");
    }
}

std::string SvgDocument::str() const {
    return std::format(
        R"(<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {0} {1}" width="{0}" height="{1}">)"
        "\n{2}</svg>\n",
        width_, height_, body_);
}

void SvgDocument::save(const std::string& filename) const {
    std::ofstream file(filename);

    if (!file) {
        throw std::runtime_error("Failed writing to file " + filename);
    }

    file << str();
    file.close();

    if (!file) {
        throw std::runtime_error("Failed writing to file " + filename);
    }
}

void SvgDocument::line(double x1, double y1, double x2, double y2, const std::string& stroke,
                       double stroke_width) {
    if (!std::isfinite(x1) || !std::isfinite(y1) || !std::isfinite(x2) || !std::isfinite(y2) ||
        !std::isfinite(stroke_width) || stroke_width < 0) {
        throw std::invalid_argument("Line coordinates must be finite and stroke width "
                                    "must be finite and non-negative");
    }

    std::format_to(std::back_inserter(body_),
                   "\t"
                   R"(<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="{}"/>)"
                   "\n",
                   x1, y1, x2, y2, escapeXml(stroke), stroke_width);
}

void SvgDocument::rect(double x, double y, double width, double height, const std::string& fill) {
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(width) || !std::isfinite(height) ||
        width < 0 || height < 0) {
        throw std::invalid_argument("Rectangle coordinates must be finite and dimensions "
                                    "must be finite and non-negative");
    }

    std::format_to(std::back_inserter(body_),
                   "\t"
                   R"(<rect x="{}" y="{}" width="{}" height="{}" fill="{}"/>)"
                   "\n",
                   x, y, width, height, escapeXml(fill));
}

void SvgDocument::polyline(std::span<const Point> points, const std::string& stroke,
                           double stroke_width) {
    if (points.size() < 2) {
        throw std::invalid_argument("Polyline requires at least 2 points");
    }

    if (!std::isfinite(stroke_width) || stroke_width < 0) {
        throw std::invalid_argument("Stroke width must be finite and non-negative");
    }

    for (const auto& point : points) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
            throw std::invalid_argument("Coordinates must be finite");
        }
    }

    std::string coordinates;

    for (const auto& point : points) {
        if (!coordinates.empty()) {
            coordinates += ' ';
        }

        std::format_to(std::back_inserter(coordinates), "{},{}", point.x, point.y);
    }

    std::format_to(
        std::back_inserter(body_),
        "\t"
        R"(<polyline points="{}" fill="none" stroke="{}" stroke-width="{}" stroke-linejoin="round"/>)"
        "\n",
        coordinates, escapeXml(stroke), stroke_width);
}

void SvgDocument::circle(double cx, double cy, double radius, const std::string& fill,
                         const std::string& stroke, double stroke_width) {
    if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(radius) ||
        !std::isfinite(stroke_width)) {
        throw std::invalid_argument("Numerical arguments must be finite");
    }

    if (radius < 0 || stroke_width < 0) {
        throw std::invalid_argument("Radius and stroke width must be non-negative");
    }

    std::format_to(std::back_inserter(body_),
                   "\t"
                   R"(<circle cx="{}" cy="{}" r="{}" fill="{}" stroke="{}" stroke-width="{}"/>)"
                   "\n",
                   cx, cy, radius, escapeXml(fill), escapeXml(stroke), stroke_width);
}

void SvgDocument::text(double x, double y, const std::string& content, double font_size,
                       const std::string& fill, TextAnchor anchor, double rotation) {
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(font_size) ||
        !std::isfinite(rotation)) {
        throw std::invalid_argument("Numerical arguments must be finite");
    }

    if (font_size <= 0) {
        throw std::invalid_argument("Font size must be positive");
    }

    const auto text_anchor = toSvgAnchor(anchor);

    std::format_to(
        std::back_inserter(body_),
        "\t"
        R"(<text x="{}" y="{}" font-size="{}" font-family="sans-serif" fill="{}" text-anchor="{}")",
        x, y, font_size, escapeXml(fill), text_anchor);
    if (rotation != 0.0) {
        std::format_to(std::back_inserter(body_), " transform=\"rotate({} {} {})\"", rotation, x,
                       y);
    }
    std::format_to(std::back_inserter(body_),
                   R"(>{}</text>)"
                   "\n",
                   escapeXml(content));
}

} // namespace neuro::viz
