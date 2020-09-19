#include "boost/gil/image_processing/numeric.hpp"
#include <boost/gil.hpp>
#include <boost/gil/extension/io/png.hpp>

namespace gil = boost::gil;

int main(int argc, char* argv[])
{
    gil::gray8_image_t input_image;
    gil::read_image(argv[1], input_image, gil::png_tag{});

    auto input = gil::view(input_image);
    auto sobel_x = gil::generate_dx_sobel();
    auto sobel_y = gil::generate_dy_sobel();
    gil::gray16_image_t dx_image(input_image.dimensions());
    gil::gray16_image_t dy_image(input_image.dimensions());
    auto dx = gil::view(dx_image);
    auto dy = gil::view(dy_image);
    gil::detail::convolve_2d(input, sobel_x, dx);
    gil::detail::convolve_2d(input, sobel_y, dy);
    gil::gray16_image_t gradient_image(input_image.dimensions());
    auto gradient = gil::view(gradient_image);
    gil::compute_gradient_strength(dx, dy, gradient);
    gil::write_view("gradient.png", gradient, gil::png_tag{});
}
