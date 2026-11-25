
#include "neuro/viz/SvgDocument.hpp"

#include <array>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <limits>
#include <locale>
#include <stdexcept>
#include <string>
#include <vector>

using namespace neuro::viz;

namespace {

constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
constexpr double Inf = std::numeric_limits<double>::infinity();

void expectContains(const std::string& document, const std::string& expected) {
    EXPECT_NE(document.find(expected), std::string::npos)
        << "Expected: " << expected << "\nDocument:\n"
        << document;
}

template <typename Function> void expectRejected(SvgDocument& svg, Function function) {
    const auto before = svg.str();

    EXPECT_THROW(function(), std::invalid_argument);
    EXPECT_EQ(svg.str(), before);
}

class CommaDecimalPoint : public std::numpunct<char> {
protected:
    char do_decimal_point() const override {
        return ',';
    }
};

class LocaleGuard {
public:
    LocaleGuard() : previous_(std::locale()) {}

    ~LocaleGuard() {
        std::locale::global(previous_);
    }

    LocaleGuard(const LocaleGuard&) = delete;
    LocaleGuard& operator=(const LocaleGuard&) = delete;

private:
    std::locale previous_;
};

class SvgFileTest : public ::testing::Test {
protected:
    std::filesystem::path path_;

    void SetUp() override {
        path_ =
            std::filesystem::temp_directory_path() /
            ("neuro_svg_" +
             std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()) + ".svg");
    }

    void TearDown() override {
        std::error_code error;
        std::filesystem::remove(path_, error);
    }

    std::string readFile() const {
        std::ifstream file(path_, std::ios::binary);

        EXPECT_TRUE(file.is_open());

        return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    }
};

} // namespace

// ============================================================
// Document construction
// ============================================================

TEST(SvgDocumentTest, EmptyDocument) {
    SvgDocument svg(400, 300);

    EXPECT_EQ(
        svg.str(),
        R"(<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300" width="400" height="300">)"
        "\n</svg>\n");
}

TEST(SvgDocumentTest, NonSquareDimensions) {
    SvgDocument svg(800, 250);

    expectContains(svg.str(), R"(viewBox="0 0 800 250")");
    expectContains(svg.str(), R"(width="800")");
    expectContains(svg.str(), R"(height="250")");
}

TEST(SvgDocumentTest, MinimumDimensions) {
    EXPECT_NO_THROW(SvgDocument(1, 1));
}

TEST(SvgDocumentTest, RejectsZeroWidth) {
    EXPECT_THROW(SvgDocument(0, 100), std::invalid_argument);
}

TEST(SvgDocumentTest, RejectsZeroHeight) {
    EXPECT_THROW(SvgDocument(100, 0), std::invalid_argument);
}

TEST(SvgDocumentTest, RejectsNegativeWidth) {
    EXPECT_THROW(SvgDocument(-100, 100), std::invalid_argument);
}

TEST(SvgDocumentTest, RejectsNegativeHeight) {
    EXPECT_THROW(SvgDocument(100, -100), std::invalid_argument);
}

// ============================================================
// Line
// ============================================================

TEST(SvgDocumentTest, LineSerialization) {
    SvgDocument svg(400, 300);

    svg.line(10, 20, 100, 200, "blue", 2.5);

    expectContains(svg.str(),
                   R"(<line x1="10" y1="20" x2="100" y2="200" stroke="blue" stroke-width="2.5"/>)");
}

TEST(SvgDocumentTest, LineDefaultStrokeWidth) {
    SvgDocument svg(400, 300);

    svg.line(0, 0, 100, 100, "black");

    expectContains(svg.str(), R"(stroke-width="1")");
}

TEST(SvgDocumentTest, LineAllowsNegativeCoordinates) {
    SvgDocument svg(400, 300);

    svg.line(-10, -20, -30, -40, "black");

    expectContains(svg.str(), R"(x1="-10" y1="-20" x2="-30" y2="-40")");
}

TEST(SvgDocumentTest, LineAllowsCoordinatesOutsideCanvas) {
    SvgDocument svg(100, 100);

    EXPECT_NO_THROW(svg.line(200, 300, 400, 500, "black"));
}

TEST(SvgDocumentTest, LineAllowsZeroStrokeWidth) {
    SvgDocument svg(400, 300);

    svg.line(0, 0, 100, 100, "black", 0);

    expectContains(svg.str(), R"(stroke-width="0")");
}

TEST(SvgDocumentTest, LineRejectsNegativeStrokeWidth) {
    SvgDocument svg(400, 300);

    expectRejected(svg, [&] { svg.line(0, 0, 100, 100, "black", -1); });
}

TEST(SvgDocumentTest, LineRejectsNonFiniteArguments) {
    SvgDocument svg(400, 300);

    for (double invalid : {NaN, Inf, -Inf}) {
        for (std::size_t i = 0; i < 5; ++i) {
            SCOPED_TRACE(i);

            std::array<double, 5> args{10, 20, 30, 40, 1};
            args[i] = invalid;

            expectRejected(svg,
                           [&] { svg.line(args[0], args[1], args[2], args[3], "black", args[4]); });
        }
    }
}

// ============================================================
// Rectangle
// ============================================================

TEST(SvgDocumentTest, RectangleSerialization) {
    SvgDocument svg(400, 300);

    svg.rect(10, 20, 100, 50, "#2563eb");

    expectContains(svg.str(), R"(<rect x="10" y="20" width="100" height="50" fill="#2563eb"/>)");
}

TEST(SvgDocumentTest, RectangleAllowsNegativePosition) {
    SvgDocument svg(400, 300);

    svg.rect(-10, -20, 100, 50, "white");

    expectContains(svg.str(), R"(x="-10" y="-20")");
}

TEST(SvgDocumentTest, RectangleAllowsZeroWidth) {
    SvgDocument svg(400, 300);

    EXPECT_NO_THROW(svg.rect(0, 0, 0, 100, "white"));
}

TEST(SvgDocumentTest, RectangleAllowsZeroHeight) {
    SvgDocument svg(400, 300);

    EXPECT_NO_THROW(svg.rect(0, 0, 100, 0, "white"));
}

TEST(SvgDocumentTest, RectangleRejectsNegativeWidth) {
    SvgDocument svg(400, 300);

    expectRejected(svg, [&] { svg.rect(0, 0, -100, 50, "white"); });
}

TEST(SvgDocumentTest, RectangleRejectsNegativeHeight) {
    SvgDocument svg(400, 300);

    expectRejected(svg, [&] { svg.rect(0, 0, 100, -50, "white"); });
}

TEST(SvgDocumentTest, RectangleRejectsNonFiniteArguments) {
    SvgDocument svg(400, 300);

    for (double invalid : {NaN, Inf, -Inf}) {
        for (std::size_t i = 0; i < 4; ++i) {
            SCOPED_TRACE(i);

            std::array<double, 4> args{10, 20, 100, 50};
            args[i] = invalid;

            expectRejected(svg, [&] { svg.rect(args[0], args[1], args[2], args[3], "white"); });
        }
    }
}

// ============================================================
// Polyline
// ============================================================

TEST(SvgDocumentTest, PolylineSerialization) {
    SvgDocument svg(400, 300);

    const std::vector<Point> points{{10, 20}, {30, 40}, {50, 60}};

    svg.polyline(points, "blue", 2.5);

    expectContains(
        svg.str(),
        R"(<polyline points="10,20 30,40 50,60" fill="none" stroke="blue" stroke-width="2.5" stroke-linejoin="round"/>)");
}

TEST(SvgDocumentTest, PolylineAcceptsTwoPoints) {
    SvgDocument svg(400, 300);

    const std::array<Point, 2> points{{{0, 0}, {100, 100}}};

    EXPECT_NO_THROW(svg.polyline(points, "black"));
}

TEST(SvgDocumentTest, PolylineDefaultStrokeWidth) {
    SvgDocument svg(400, 300);

    const std::vector<Point> points{{0, 0}, {10, 10}};

    svg.polyline(points, "black");

    expectContains(svg.str(), R"(stroke-width="1")");
}

TEST(SvgDocumentTest, PolylinePreservesPointOrder) {
    SvgDocument svg(400, 300);

    const std::vector<Point> points{{100, 20}, {50, 90}, {75, 10}, {0, 30}};

    svg.polyline(points, "black");

    expectContains(svg.str(), R"(points="100,20 50,90 75,10 0,30")");
}

TEST(SvgDocumentTest, PolylineSupportsFractionalCoordinates) {
    SvgDocument svg(400, 300);

    const std::vector<Point> points{{1.25, 2.5}, {3.75, 4.125}};

    svg.polyline(points, "black");

    expectContains(svg.str(), R"(points="1.25,2.5 3.75,4.125")");
}

TEST(SvgDocumentTest, PolylineSupportsNegativeCoordinates) {
    SvgDocument svg(400, 300);

    const std::vector<Point> points{{-10, -20}, {-30, -40}};

    svg.polyline(points, "black");

    expectContains(svg.str(), R"(points="-10,-20 -30,-40")");
}

TEST(SvgDocumentTest, PolylineSupportsRepeatedPoints) {
    SvgDocument svg(400, 300);

    const std::vector<Point> points{{10, 20}, {10, 20}, {30, 40}};

    svg.polyline(points, "black");

    expectContains(svg.str(), R"(points="10,20 10,20 30,40")");
}

TEST(SvgDocumentTest, PolylineRejectsEmptyPoints) {
    SvgDocument svg(400, 300);

    const std::vector<Point> points;

    expectRejected(svg, [&] { svg.polyline(points, "black"); });
}

TEST(SvgDocumentTest, PolylineRejectsSinglePoint) {
    SvgDocument svg(400, 300);

    const std::vector<Point> points{{10, 20}};

    expectRejected(svg, [&] { svg.polyline(points, "black"); });
}

TEST(SvgDocumentTest, PolylineAllowsZeroStrokeWidth) {
    SvgDocument svg(400, 300);

    const std::vector<Point> points{{0, 0}, {10, 10}};

    svg.polyline(points, "black", 0);

    expectContains(svg.str(), R"(stroke-width="0")");
}

TEST(SvgDocumentTest, PolylineRejectsInvalidStrokeWidth) {
    SvgDocument svg(400, 300);

    const std::vector<Point> points{{0, 0}, {10, 10}};

    for (double invalid : {-1.0, NaN, Inf, -Inf}) {
        expectRejected(svg, [&] { svg.polyline(points, "black", invalid); });
    }
}

TEST(SvgDocumentTest, PolylineRejectsNonFiniteCoordinates) {
    SvgDocument svg(400, 300);

    for (double invalid : {NaN, Inf, -Inf}) {
        const std::vector<Point> invalid_x{{0, 0}, {invalid, 10}};

        const std::vector<Point> invalid_y{{0, 0}, {10, invalid}};

        expectRejected(svg, [&] { svg.polyline(invalid_x, "black"); });

        expectRejected(svg, [&] { svg.polyline(invalid_y, "black"); });
    }
}

TEST(SvgDocumentTest, PolylineLateInvalidPointPreservesDocument) {
    SvgDocument svg(400, 300);

    svg.rect(0, 0, 400, 300, "white");

    const std::vector<Point> points{{10, 20}, {30, 40}, {50, 60}, {NaN, 80}};

    expectRejected(svg, [&] { svg.polyline(points, "blue"); });

    svg.line(0, 0, 100, 100, "black");

    expectContains(svg.str(), "<rect ");
    expectContains(svg.str(), "<line ");
    EXPECT_EQ(svg.str().find("<polyline "), std::string::npos);
}

// ============================================================
// Circle
// ============================================================

TEST(SvgDocumentTest, CircleSerialization) {
    SvgDocument svg(400, 300);

    svg.circle(100, 80, 6, "blue", "white", 1.5);

    expectContains(
        svg.str(),
        R"(<circle cx="100" cy="80" r="6" fill="blue" stroke="white" stroke-width="1.5"/>)");
}

TEST(SvgDocumentTest, CircleDefaultStroke) {
    SvgDocument svg(400, 300);

    svg.circle(100, 80, 6, "blue");

    expectContains(svg.str(), R"(stroke="none")");
    expectContains(svg.str(), R"(stroke-width="1")");
}

TEST(SvgDocumentTest, CircleAllowsNegativeCenter) {
    SvgDocument svg(400, 300);

    svg.circle(-100, -80, 6, "blue");

    expectContains(svg.str(), R"(cx="-100" cy="-80")");
}

TEST(SvgDocumentTest, CircleAllowsZeroRadius) {
    SvgDocument svg(400, 300);

    svg.circle(100, 80, 0, "blue");

    expectContains(svg.str(), R"(r="0")");
}

TEST(SvgDocumentTest, CircleAllowsZeroStrokeWidth) {
    SvgDocument svg(400, 300);

    svg.circle(100, 80, 6, "blue", "white", 0);

    expectContains(svg.str(), R"(stroke-width="0")");
}

TEST(SvgDocumentTest, CircleRejectsNegativeRadius) {
    SvgDocument svg(400, 300);

    expectRejected(svg, [&] { svg.circle(100, 80, -6, "blue"); });
}

TEST(SvgDocumentTest, CircleRejectsNegativeStrokeWidth) {
    SvgDocument svg(400, 300);

    expectRejected(svg, [&] { svg.circle(100, 80, 6, "blue", "white", -1); });
}

TEST(SvgDocumentTest, CircleRejectsNonFiniteArguments) {
    SvgDocument svg(400, 300);

    for (double invalid : {NaN, Inf, -Inf}) {
        for (std::size_t i = 0; i < 4; ++i) {
            SCOPED_TRACE(i);

            std::array<double, 4> args{100, 80, 6, 1};
            args[i] = invalid;

            expectRejected(
                svg, [&] { svg.circle(args[0], args[1], args[2], "blue", "white", args[3]); });
        }
    }
}

// ============================================================
// Text
// ============================================================

TEST(SvgDocumentTest, TextSerialization) {
    SvgDocument svg(400, 300);

    svg.text(100, 50, "Training Loss", 18, "#334155", TextAnchor::Middle);

    expectContains(
        svg.str(),
        R"(<text x="100" y="50" font-size="18" font-family="sans-serif" fill="#334155" text-anchor="middle">Training Loss</text>)");
}

TEST(SvgDocumentTest, TextDefaultArguments) {
    SvgDocument svg(400, 300);

    svg.text(100, 50, "Loss");

    expectContains(
        svg.str(),
        R"(<text x="100" y="50" font-size="12" font-family="sans-serif" fill="black" text-anchor="start">Loss</text>)");
}

TEST(SvgDocumentTest, TextStartAnchor) {
    SvgDocument svg(400, 300);

    svg.text(100, 50, "Loss", 12, "black", TextAnchor::Start);

    expectContains(svg.str(), R"(text-anchor="start")");
}

TEST(SvgDocumentTest, TextMiddleAnchor) {
    SvgDocument svg(400, 300);

    svg.text(100, 50, "Loss", 12, "black", TextAnchor::Middle);

    expectContains(svg.str(), R"(text-anchor="middle")");
}

TEST(SvgDocumentTest, TextEndAnchor) {
    SvgDocument svg(400, 300);

    svg.text(100, 50, "Loss", 12, "black", TextAnchor::End);

    expectContains(svg.str(), R"(text-anchor="end")");
}

TEST(SvgDocumentTest, TextAllowsEmptyContent) {
    SvgDocument svg(400, 300);

    svg.text(100, 50, "");

    expectContains(svg.str(), R"(text-anchor="start"></text>)");
}

TEST(SvgDocumentTest, TextPreservesContentWhitespace) {
    SvgDocument svg(400, 300);

    svg.text(100, 50, "  Training Loss  ");

    expectContains(svg.str(), ">  Training Loss  </text>");
}

TEST(SvgDocumentTest, TextPreservesUtf8) {
    SvgDocument svg(400, 300);

    svg.text(100, 50, "Accuracy — café");

    expectContains(svg.str(), ">Accuracy — café</text>");
}

TEST(SvgDocumentTest, TextAllowsNegativeCoordinates) {
    SvgDocument svg(400, 300);

    svg.text(-100, -50, "Loss");

    expectContains(svg.str(), R"(x="-100" y="-50")");
}

TEST(SvgDocumentTest, TextAllowsFractionalFontSize) {
    SvgDocument svg(400, 300);

    svg.text(100, 50, "Loss", 12.5);

    expectContains(svg.str(), R"(font-size="12.5")");
}

TEST(SvgDocumentTest, TextRejectsZeroFontSize) {
    SvgDocument svg(400, 300);

    expectRejected(svg, [&] { svg.text(100, 50, "Loss", 0); });
}

TEST(SvgDocumentTest, TextRejectsNegativeFontSize) {
    SvgDocument svg(400, 300);

    expectRejected(svg, [&] { svg.text(100, 50, "Loss", -12); });
}

TEST(SvgDocumentTest, TextRejectsInvalidAnchor) {
    SvgDocument svg(400, 300);

    expectRejected(svg,
                   [&] { svg.text(100, 50, "Loss", 12, "black", static_cast<TextAnchor>(99)); });
}

TEST(SvgDocumentTest, TextRejectsNonFiniteArguments) {
    SvgDocument svg(400, 300);

    for (double invalid : {NaN, Inf, -Inf}) {
        for (std::size_t i = 0; i < 3; ++i) {
            SCOPED_TRACE(i);

            std::array<double, 3> args{100, 50, 12};
            args[i] = invalid;

            expectRejected(svg, [&] { svg.text(args[0], args[1], "Loss", args[2]); });
        }
    }
}

// ============================================================
// XML escaping
// ============================================================

TEST(SvgDocumentTest, TextEscapesAllXmlCharacters) {
    SvgDocument svg(400, 300);

    svg.text(100, 50, "&<>\"'");

    expectContains(svg.str(), ">&amp;&lt;&gt;&quot;&apos;</text>");
}

TEST(SvgDocumentTest, TextEscapesAmpersand) {
    SvgDocument svg(400, 300);

    svg.text(100, 50, "Training & Validation Loss");

    expectContains(svg.str(), ">Training &amp; Validation Loss</text>");
}

TEST(SvgDocumentTest, TextEscapesXmlMarkup) {
    SvgDocument svg(400, 300);

    svg.text(100, 50, "<rect/>");

    expectContains(svg.str(), ">&lt;rect/&gt;</text>");
    EXPECT_EQ(svg.str().find("<rect/>"), std::string::npos);
}

TEST(SvgDocumentTest, TextEscapesRepeatedCharacters) {
    SvgDocument svg(400, 300);

    svg.text(100, 50, "&&<<>>");

    expectContains(svg.str(), ">&amp;&amp;&lt;&lt;&gt;&gt;</text>");
}

TEST(SvgDocumentTest, TextTreatsExistingEntitiesAsLiteralText) {
    SvgDocument svg(400, 300);

    svg.text(100, 50, "&amp;");

    expectContains(svg.str(), ">&amp;amp;</text>");
}

TEST(SvgDocumentTest, TextEscapesFillAttribute) {
    SvgDocument svg(400, 300);

    svg.text(100, 50, "Loss", 12, "&<>\"'");

    expectContains(svg.str(), R"(fill="&amp;&lt;&gt;&quot;&apos;")");
}

TEST(SvgDocumentTest, LineEscapesStrokeAttribute) {
    SvgDocument svg(400, 300);

    svg.line(0, 0, 100, 100, "&<>\"'");

    expectContains(svg.str(), R"(stroke="&amp;&lt;&gt;&quot;&apos;")");
}

TEST(SvgDocumentTest, RectangleEscapesFillAttribute) {
    SvgDocument svg(400, 300);

    svg.rect(0, 0, 100, 100, "&<>\"'");

    expectContains(svg.str(), R"(fill="&amp;&lt;&gt;&quot;&apos;")");
}

TEST(SvgDocumentTest, PolylineEscapesStrokeAttribute) {
    SvgDocument svg(400, 300);

    const std::vector<Point> points{{0, 0}, {10, 10}};

    svg.polyline(points, "&<>\"'");

    expectContains(svg.str(), R"(stroke="&amp;&lt;&gt;&quot;&apos;")");
}

TEST(SvgDocumentTest, CircleEscapesFillAndStrokeAttributes) {
    SvgDocument svg(400, 300);

    svg.circle(100, 80, 6, "&<>\"'", "&<>\"'");

    expectContains(svg.str(), R"(fill="&amp;&lt;&gt;&quot;&apos;")");

    expectContains(svg.str(), R"(stroke="&amp;&lt;&gt;&quot;&apos;")");
}

// ============================================================
// Document integrity and drawing order
// ============================================================

TEST(SvgDocumentTest, PreservesDrawingOrder) {
    SvgDocument svg(400, 300);

    const std::vector<Point> points{{0, 0}, {10, 10}};

    svg.rect(0, 0, 400, 300, "white");
    svg.line(0, 0, 100, 100, "black");
    svg.polyline(points, "blue");
    svg.circle(10, 10, 3, "blue");
    svg.text(100, 50, "Loss");

    const auto document = svg.str();

    const auto rect = document.find("<rect ");
    const auto line = document.find("<line ");
    const auto polyline = document.find("<polyline ");
    const auto circle = document.find("<circle ");
    const auto text = document.find("<text ");

    ASSERT_NE(rect, std::string::npos);
    ASSERT_NE(line, std::string::npos);
    ASSERT_NE(polyline, std::string::npos);
    ASSERT_NE(circle, std::string::npos);
    ASSERT_NE(text, std::string::npos);

    EXPECT_LT(rect, line);
    EXPECT_LT(line, polyline);
    EXPECT_LT(polyline, circle);
    EXPECT_LT(circle, text);
}

TEST(SvgDocumentTest, RepeatedSerializationIsIdentical) {
    SvgDocument svg(400, 300);

    svg.rect(0, 0, 400, 300, "white");
    svg.text(100, 50, "Loss");

    const auto first = svg.str();

    EXPECT_EQ(svg.str(), first);
    EXPECT_EQ(svg.str(), first);
}

TEST(SvgDocumentTest, CanDrawAfterSerialization) {
    SvgDocument svg(400, 300);

    svg.rect(0, 0, 400, 300, "white");

    const auto before = svg.str();

    svg.circle(100, 80, 6, "blue");

    const auto after = svg.str();

    EXPECT_NE(before, after);
    expectContains(after, "<rect ");
    expectContains(after, "<circle ");

    EXPECT_TRUE(after.ends_with("</svg>\n"));
}

TEST(SvgDocumentTest, ContainsExactlyOneRootElement) {
    SvgDocument svg(400, 300);

    svg.line(0, 0, 100, 100, "black");
    svg.text(100, 50, "Loss");

    const auto document = svg.str();

    EXPECT_TRUE(document.starts_with("<svg "));
    EXPECT_TRUE(document.ends_with("</svg>\n"));

    EXPECT_EQ(document.find("<svg ", 1), std::string::npos);

    EXPECT_EQ(document.find("</svg>"), document.rfind("</svg>"));
}

TEST(SvgDocumentTest, IndependentDocuments) {
    SvgDocument first(400, 300);
    SvgDocument second(800, 600);

    first.text(100, 50, "First");
    second.text(200, 100, "Second");

    expectContains(first.str(), ">First</text>");
    expectContains(second.str(), ">Second</text>");

    EXPECT_EQ(first.str().find("Second"), std::string::npos);
    EXPECT_EQ(second.str().find("First"), std::string::npos);
}

// ============================================================
// Numerical formatting
// ============================================================

TEST(SvgDocumentTest, FractionalCoordinatesArePreserved) {
    SvgDocument svg(400, 300);

    svg.line(1.25, 2.5, 3.75, 4.125, "black", 0.5);

    expectContains(svg.str(), R"(x1="1.25" y1="2.5" x2="3.75" y2="4.125")");

    expectContains(svg.str(), R"(stroke-width="0.5")");
}

TEST(SvgDocumentTest, PreservesDoublePrecision) {
    SvgDocument svg(400, 300);

    svg.line(1.23456789012345, 0, 100, 100, "black");

    expectContains(svg.str(), R"(x1="1.23456789012345")");
}

TEST(SvgDocumentTest, FormattingIsLocaleIndependent) {
    LocaleGuard guard;

    std::locale::global(std::locale(std::locale::classic(), new CommaDecimalPoint));

    SvgDocument svg(400, 300);

    const std::vector<Point> points{{1.25, 2.5}, {3.75, 4.125}};

    svg.line(1.25, 2.5, 3.75, 4.125, "black", 0.5);
    svg.rect(1.25, 2.5, 3.75, 4.125, "white");
    svg.polyline(points, "blue", 0.5);
    svg.circle(1.25, 2.5, 3.75, "blue", "white", 0.5);
    svg.text(1.25, 2.5, "Loss", 12.5);

    const auto document = svg.str();

    expectContains(document, R"(x1="1.25")");
    expectContains(document, R"(width="3.75")");
    expectContains(document, R"(points="1.25,2.5 3.75,4.125")");
    expectContains(document, R"(cx="1.25")");
    expectContains(document, R"(font-size="12.5")");

    EXPECT_EQ(document.find("\"1,25\""), std::string::npos);
    EXPECT_EQ(document.find("\"12,5\""), std::string::npos);
}

// ============================================================
// File output
// ============================================================

TEST_F(SvgFileTest, SaveCreatesFile) {
    SvgDocument svg(400, 300);

    svg.save(path_.string());

    EXPECT_TRUE(std::filesystem::exists(path_));
}

TEST_F(SvgFileTest, SavedContentMatchesSerialization) {
    SvgDocument svg(400, 300);

    svg.rect(0, 0, 400, 300, "white");
    svg.line(0, 0, 100, 100, "black");
    svg.circle(100, 100, 5, "blue");
    svg.text(100, 50, "Training & Validation");

    svg.save(path_.string());

    EXPECT_EQ(readFile(), svg.str());
}

TEST_F(SvgFileTest, SaveOverwritesExistingFile) {
    SvgDocument svg(400, 300);

    {
        std::ofstream file(path_);
        ASSERT_TRUE(file.is_open());
        file << std::string(10000, 'x');
    }

    svg.save(path_.string());

    EXPECT_EQ(readFile(), svg.str());
}

TEST_F(SvgFileTest, CanSaveUpdatedDocument) {
    SvgDocument svg(400, 300);

    svg.rect(0, 0, 400, 300, "white");
    svg.save(path_.string());

    const auto first = readFile();

    svg.text(100, 50, "Loss");
    svg.save(path_.string());

    EXPECT_NE(readFile(), first);
    EXPECT_EQ(readFile(), svg.str());
}

TEST_F(SvgFileTest, SaveDoesNotModifyDocument) {
    SvgDocument svg(400, 300);

    svg.text(100, 50, "Loss");

    const auto before = svg.str();

    svg.save(path_.string());

    EXPECT_EQ(svg.str(), before);
}

TEST_F(SvgFileTest, SaveRejectsInvalidParentPath) {
    SvgDocument svg(400, 300);

    // A regular file cannot act as a parent directory.
    {
        std::ofstream file(path_);
        ASSERT_TRUE(file.is_open());
    }

    const auto invalid_path = path_ / "plot.svg";

    EXPECT_THROW(svg.save(invalid_path.string()), std::runtime_error);
}

TEST(SvgDocumentTest, SaveRejectsEmptyFilename) {
    SvgDocument svg(400, 300);

    EXPECT_THROW(svg.save(""), std::runtime_error);
}

TEST(SvgDocumentTest, SaveRejectsDirectoryPath) {
    SvgDocument svg(400, 300);

    EXPECT_THROW(svg.save(std::filesystem::temp_directory_path().string()), std::runtime_error);
}