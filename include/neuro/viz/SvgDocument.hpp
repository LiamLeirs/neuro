
#pragma once

#include <span>
#include <string>

namespace neuro::viz {

struct Point {
    double x;
    double y;
};

enum class TextAnchor { Start, Middle, End };

class SvgDocument {
public:
    SvgDocument(int width, int height);

    std::string str() const;
    void save(const std::string& filename) const;

    void line(double x1, double y1, double x2, double y2, const std::string& stroke,
              double stroke_width = 1.0);

    void rect(double x, double y, double width, double height, const std::string& fill);

    void polyline(std::span<const Point> points, const std::string& stroke,
                  double stroke_width = 1.0);

    void circle(double cx, double cy, double radius, const std::string& fill,
                const std::string& stroke = "none", double stroke_width = 1.0);

    void text(double x, double y, const std::string& content, double font_size = 12.0,
              const std::string& fill = "black", TextAnchor anchor = TextAnchor::Start);

private:
    int width_;
    int height_;
    std::string body_;
};

} // namespace neuro::viz
