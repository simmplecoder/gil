#include <boost/gil/image_view_factory.hpp>
#include <algorithm>
#include <boost/gil.hpp>
#include <boost/gil/extension/io/png.hpp>

namespace gil = boost::gil;

enum class bidirection
{
    main_diagonal,
    vertical,
    second_diagonal,
    horizontal
};

struct orthogonal_gradient_extractor
{
    gil::gray32f_view_t gradient_angle;
    bidirection to_bidirection(float angle) const
    {
        const auto pi = static_cast<float>(gil::detail::pi);
        const auto one_eightth_pi = pi / 8.0;
        // use the fact that bidirection is symmetrical around origin
        if (angle < 0)
        {
            angle += pi;
        }

        if (angle > 5 * one_eightth_pi && angle <= 7 * one_eightth_pi)
        {
            return bidirection::main_diagonal;
        }
        else if (angle > 3 * one_eightth_pi && angle <= 5 * one_eightth_pi)
        {
            return bidirection::vertical;
        }
        else if (angle > one_eightth_pi && angle <= 3 * one_eightth_pi)
        {
            return bidirection::second_diagonal;
        }
        else
        {
            return bidirection::horizontal;
        }
    }

    bidirection to_orthogonal_direction(bidirection bidir) const
    {
        switch (bidir)
        {
        case bidirection::main_diagonal:
            return bidirection::second_diagonal;
        case bidirection::vertical:
            return bidirection::horizontal;
        case bidirection::second_diagonal:
            return bidirection::main_diagonal;
        default:
            return bidirection::vertical;
        }
    }

    template <typename View>
    std::array<typename View::value_type, 3> operator()(View const& window,
                                                        gil::point_t origin) const
    {
        using pixel_type = typename View::value_type;
        auto bidirection = to_orthogonal_direction(to_bidirection(gradient_angle(origin)[0]));
        std::array<pixel_type, 3> points;
        gil::point_t (*index_to_point)(std::ptrdiff_t, std::ptrdiff_t) = nullptr;
        switch (bidirection)
        {
        case bidirection::main_diagonal:
            index_to_point = [](std::ptrdiff_t i, std::ptrdiff_t /*width*/)
            {
                return gil::point_t(i, i);
            };
            break;
        case bidirection::vertical:
            index_to_point = [](std::ptrdiff_t i, std::ptrdiff_t /*width*/)
            {
                return gil::point_t(1, i);
            };
            break;
        case bidirection::second_diagonal:
            index_to_point = [](std::ptrdiff_t i, std::ptrdiff_t width)
            {
                return gil::point_t(width - 1 - i, i);
            };
            break;
        default:
            index_to_point = [](std::ptrdiff_t i, std::ptrdiff_t /*width*/)
            {
                return gil::point_t(i, 1);
            };
        }
        for (std::ptrdiff_t i = 0; i < 3; ++i)
        {
            points[i] = window(index_to_point(i, window.width()));
        }
        return points;
    }
};

void hysteresis(const gil::gray16_view_t& input,
                                     const gil::gray16_view_t& output,
                                     std::uint16_t soft_threshold,
                                     std::uint16_t hard_threshold,
                                     gil::gray16_pixel_t pass_value,
                                     gil::gray16_pixel_t fail_value) {
    enum pass_state {
        fail = 0,
        soft_pass = 1,
        pass = 2
    };

    struct cell {
        std::uint32_t parent;
        pass_state state;
    };
    std::vector<cell> cells(input.size());
    for (std::size_t i = 0; i < cells.size(); ++i) {
        cells[i].parent = i;
    }

    std::size_t width = static_cast<size_t>(input.width());
    for (std::size_t y = 0; y < input.height(); ++y) {
        for (std::size_t x = 0; x < input.width(); ++x) {
            auto flat_index = y * width + x;
            auto current_pixel = input(x, y);
            if (current_pixel < soft_threshold) {
                cells[flat_index].state = fail;
                output(x, y) = fail_value;
                continue;
            }

            cells[flat_index].state = current_pixel < hard_threshold ? soft_pass : pass;

            auto left_neighbor = flat_index - 1;
            auto upper_neighbor = flat_index - width;
            if (x > 0 && cells[flat_index].state != fail) {
                cells[flat_index].parent = cells[left_neighbor].parent;
                auto parent = cells[flat_index].parent;
                auto best_state = std::max(cells[parent].state, cells[flat_index].state);
                cells[flat_index].state = best_state;
                cells[parent].state = best_state;
            } else if (y > 0 && cells[flat_index].state != fail) {
                cells[flat_index].parent = cells[upper_neighbor].parent;
                auto parent = cells[flat_index].parent;
                auto best_state = std::max(cells[parent].state, cells[flat_index].state);
                cells[flat_index].state = best_state;
                cells[parent].state = best_state;
            }
        }
    }

    for (std::size_t y = 0; y < input.height(); ++y) {
        for (std::size_t x = 0; x < input.width(); ++x) {
            auto flat_index = y * width + x;
            if (cells[cells[flat_index].parent].state == pass) {
                output(x, y) = pass_value;
            } else {
                output(x, y) = fail_value;
            }
        }
    }
}

void gradient(const gil::gray8_view_t& input, const gil::gray16_view_t& strength, const gil::gray32f_view_t& angle) {
    auto sobel_x = gil::generate_dx_sobel();
    auto sobel_y = gil::generate_dy_sobel();
    gil::gray16s_image_t dx_image(input.dimensions());
    gil::gray16s_image_t dy_image(input.dimensions());
    auto dx = gil::view(dx_image);
    auto dy = gil::view(dy_image);
    gil::detail::convolve_2d(input, sobel_x, dx);
    gil::detail::convolve_2d(input, sobel_y, dy);

    gil::compute_gradient_strength(dx, dy, strength);

    auto max_element = *std::max_element(strength.begin(), strength.end(),
                                         [](gil::gray16_pixel_t lhs, gil::gray16_pixel_t rhs)
                                         {
                                             return lhs[0] < rhs[0];
                                         });
    gil::transform_pixels(strength, strength,
                          [max_element](gil::gray16_pixel_t pixel)
                          {
                              return (pixel[0] / static_cast<double>(max_element[0])) *
                                     std::numeric_limits<gil::uint16_t>::max();
                          });

    gil::compute_gradient_angle(dx, dy, angle);
}

int main(int argc, char* argv[])
{
    gil::gray8_image_t input_image;
    gil::read_image(argv[1], input_image, gil::png_tag{});

    auto input = gil::view(input_image);

    gil::gray16_image_t gradient_image(input_image.dimensions());
    auto gradient_strength = gil::view(gradient_image);
    gil::gray32f_image_t gradient_angle_image(input.dimensions());
    auto gradient_angle = gil::view(gradient_angle_image);
    gradient(input, gradient_strength, gradient_angle);

    gil::write_view("gradient_strength.png", gil::color_converted_view<gil::gray8_pixel_t>(gradient_strength),
                    gil::png_tag{});

    orthogonal_gradient_extractor extractor{gradient_angle};
    gil::gray16_image_t suppressed_image(input.dimensions());
    auto suppressed = gil::view(suppressed_image);
    gil::nonmax_suppression(gradient_strength, 3, gil::gray16_pixel_t(0), suppressed, extractor);
    gil::write_view("suppressed.png", gil::color_converted_view<gil::gray8_pixel_t>(suppressed),
                    gil::png_tag{});

    auto soft_threshold = std::stoul(argv[2]);
    auto hard_threshold = std::stoul(argv[3]);

    gil::gray16_image_t final_image(input.dimensions());
    auto final = gil::view(final_image);
    hysteresis(suppressed,
               final,
               static_cast<uint16_t>(soft_threshold),
               static_cast<uint16_t>(hard_threshold),
               gil::gray16_pixel_t(std::numeric_limits<std::uint16_t>::max()),
               gil::gray16_pixel_t(0));
    gil::write_view("final.png", final, gil::png_tag{});
}
