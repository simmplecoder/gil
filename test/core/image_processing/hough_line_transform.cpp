#include "boost/gil/extension/io/png/tags.hpp"
#include <algorithm>
#include <boost/core/lightweight_test.hpp>
#include <boost/gil/detail/math.hpp>
#include <boost/gil/extension/io/png.hpp>
#include <boost/gil/image.hpp>
#include <boost/gil/image_processing/hough_transform.hpp>
#include <boost/gil/image_view.hpp>
#include <boost/gil/io/write_view.hpp>
#include <boost/gil/point.hpp>
#include <boost/gil/rasterization/line.hpp>
#include <boost/gil/typedefs.hpp>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <vector>

namespace gil = boost::gil;

const std::ptrdiff_t width = 64;

void translate(std::vector<gil::point_t>& points, std::ptrdiff_t intercept)
{
    std::transform(points.begin(), points.end(), points.begin(), [intercept](gil::point_t point) {
        return gil::point_t{point.x, point.y + intercept};
    });
}

void hough_line_test(std::ptrdiff_t height, std::ptrdiff_t intercept)
{
    gil::gray8_image_t image(width, width, gil::gray8_pixel_t(0));
    auto input = gil::view(image);
    std::vector<gil::point_t> line_points(gil::estimate_point_count(width, height));
    gil::rasterize_line_bresenham(width, height, line_points.begin());
    std::cout << "line count: " << line_points.size() << '\n';
    translate(line_points, intercept);
    for (const auto& p : line_points)
    {
        input(p) = 255;
    }

    double alpha = std::atan2(height, width);
    const double theta = alpha + gil::detail::pi / 2;
    const auto minimum_angle_step = gil::minimum_angle_step({width, height});
    const double expected_alpha = std::round(alpha / minimum_angle_step) * minimum_angle_step;
    const double expected_radius = std::round(intercept * std::cos(expected_alpha));
    const double expected_theta = std::round(theta / minimum_angle_step) * minimum_angle_step;
    std::cout << "expected theta=" << theta << " and expected_radius=" << expected_radius << '\n';

    const std::size_t half_step_count = 3;
    const std::size_t expected_index = 3;
    const std::size_t accumulator_array_dimensions = half_step_count * 3 + 1;
    auto radius_param =
        gil::hough_parameter<std::ptrdiff_t>::from_step_count(expected_radius, 3, half_step_count);
    auto theta_param = gil::make_theta_parameter(
        expected_theta, minimum_angle_step * half_step_count, {width, height});
    gil::gray32_image_t accumulator_array_image(
        accumulator_array_dimensions, accumulator_array_dimensions, gil::gray32_pixel_t(0));
    auto accumulator_array = gil::view(accumulator_array_image);
    gil::hough_line_transform(input, accumulator_array, theta_param, radius_param);

    auto max_element_iterator =
        std::max_element(accumulator_array.begin(), accumulator_array.end());
    // std::cout << *max_element_iterator << ' ' << accumulator_array({expected_index,
    // expected_index})
    //           << '\n';
    // BOOST_TEST(*max_element_iterator == accumulator_array({expected_index, expected_index}));
    gil::point_t candidates[] = {
        {expected_index - 1, expected_index - 1}, {expected_index, expected_index - 1},
        {expected_index + 1, expected_index - 1}, {expected_index - 1, expected_index},
        {expected_index, expected_index},         {expected_index + 1, expected_index},
        {expected_index - 1, expected_index + 1}, {expected_index, expected_index + 1},
        {expected_index + 1, expected_index + 1}};
    bool match_found = false;
    for (std::size_t i = 0; i < 9; ++i)
    {
        if (*max_element_iterator == accumulator_array(candidates[i]))
        {
            match_found = true;
            break;
        }
    }
    BOOST_TEST(match_found);
    std::ostringstream oss;
    oss << "test-height" << height << "-intercept" << intercept << ".png";
    gil::write_view(oss.str(), input, gil::png_tag{});

    for (std::ptrdiff_t radius_index = 0; radius_index < accumulator_array_dimensions;
         ++radius_index)
    {
        for (std::ptrdiff_t theta_index = 0; theta_index < accumulator_array_dimensions;
             ++theta_index)
        {
            const auto current_radius =
                radius_param.start_point + radius_index * radius_param.step_size;
            const auto current_theta =
                theta_param.start_point + theta_index * theta_param.step_size;
            std::cout << "radius=" << current_radius << ", theta=" << current_theta
                      << ", value=" << accumulator_array(theta_index, radius_index)[0] << " index=("
                      << theta_index << ", " << radius_index << ")\n";
        }
    }
}

int main()
{
    for (std::ptrdiff_t height = 1; height < width; ++height)
    {
        hough_line_test(height, width - height - 1);
    }
    return boost::report_errors();
}