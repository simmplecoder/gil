#include "boost/gil/algorithm.hpp"
#include <functional>
#include <memory>
#include <boost/gil/color_base_algorithm.hpp>
#include <boost/gil/extension/numeric/convolve.hpp>
#include <boost/gil/extension/numeric/kernel.hpp>

namespace boost {namespace gil {
template <typename T = float, typename Allocator = std::allocator<T>>
inline detail::kernel_2d<T, Allocator> laplacian_5point_stencil() {
    std::array<T, 9> stencil {
        0, 1, 0, 1, -4, 1, 0, 1, 0
    };
    return {stencil};
}

template <typename T = float, typename Allocator = std::allocator<T>>
inline detail::kernel_2d<T, Allocator> laplacian_9point_stencil() {
    std::array<T, 9> stencil {
        0.25, 0.5, 0.25, 0.5, -3, 0.5, 0.25, 0.5, 0.25
    };
    return {stencil};
}

template <typename T = float, typename Allocator = std::allocator<T>>
inline detail::kernel_2d<T, Allocator> laplacian_9point_stencil_equal() {
    std::array<T, 9> stencil {
        1.0, 1.0, 1.0, 1.0, -8.0, 1.0, 1.0, 1.0, 1.0
    };
    return {stencil};
}

template <typename InputView, typename Kernel, typename OutputView>
void laplacian_direct(const InputView& input, const Kernel& dx_operator, const Kernel& dy_operator, OutputView dx_2, OutputView dy_2) {
    using pixel_type = typename OutputView::value_type;
    pixel_type zero_pixel;
    static_fill(zero_pixel, 0);
    image<pixel_type> dx_image(input.dimensions(), zero_pixel);
    auto dx = view(dx_image);
    image<pixel_type> dy_image(input.dimensions(), zero_pixel);
    auto dy = view(dy_image);

    detail::convolve_2d(input, dx_operator, dx);
    detail::convolve_2d(input, dy_operator, dy);
    detail::convolve_2d(dx, dx_operator, dx_2);
    detail::convolve_2d(dy, dy_operator, dy_2);
}

template <typename InputView, typename Kernel, typename OutputView>
void laplacian_direct(const InputView& input, const Kernel& dx_operator, const Kernel& dy_operator, OutputView output) {
    using pixel_type = typename OutputView::value_type;
    pixel_type zero_pixel;
    static_fill(zero_pixel, 0);
    image<pixel_type> dx_2_image(input.dimensions(), zero_pixel);
    image<pixel_type> dy_2_image(input.dimensions(), zero_pixel);
    laplacian_direct(input, dx_operator, dy_operator, view(dx_2_image), view(dy_2_image));
    transform_pixels(view(dx_2_image), view(dy_2_image), output, [](const pixel_type& dx_2, const pixel_type& dy_2) {
        pixel_type result;
        static_transform(dx_2, dy_2, result, std::plus<typename channel_type<pixel_type>::type>{});
    });
}


}}
