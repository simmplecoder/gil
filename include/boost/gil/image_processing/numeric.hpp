//
// Copyright 2019 Olzhas Zhumabek <anonymous.from.applecity@gmail.com>
//
// Use, modification and distribution are subject to the Boost Software License,
// Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
#ifndef BOOST_GIL_IMAGE_PROCESSING_NUMERIC_HPP
#define BOOST_GIL_IMAGE_PROCESSING_NUMERIC_HPP

#include <boost/gil/detail/math.hpp>
#include <cmath>

namespace boost { namespace gil {

/// \defgroup ImageProcessingMath
/// \brief Math operations for IP algorithms
///
/// This is mostly handful of mathemtical operations that are required by other
/// image processing algorithms
///
/// \brief Normalized cardinal sine
/// \ingroup ImageProcessingMath
///
/// normalized_sinc(x) = sin(pi * x) / (pi * x)
///
inline double normalized_sinc(double x)
{
    return std::sin(x * boost::gil::pi) / (x * boost::gil::pi);
}

/// \brief Lanczos response at point x
/// \ingroup ImageProcessingMath
///
/// Lanczos response is defined as:
/// x == 0: 1
/// -a < x && x < a: 0
/// otherwise: normalized_sinc(x) / normalized_sinc(x / a)
inline double lanczos(double x, std::ptrdiff_t a)
{
    if (x == 0)
        return 1;

    if (-a < x && x < a)
        return normalized_sinc(x) / normalized_sinc(x / static_cast<double>(a));

    return 0;
}

/// Fills supplied view with normalized mean
/// in which all entries will be equal to
/// 1 / (dst.size())
void generate_normalized_mean(boost::gil::gray32f_view_t dst) {
    const float entry = 1.0f / dst.size();

    for (auto& pixel: dst) {
        pixel.at(std::integral_constant<int, 0>{}) = entry;
    }
}

/// Fills supplied view with 1s (ones)
void generate_unnormalized_mean(boost::gil::gray32f_view_t dst) {
    for (auto& pixel: dst) {
        pixel.at(std::integral_constant<int, 0>{}) = 1.0f;
    }
}

/// Fills supplied view with values taken from Gaussian distribution. See
/// https://en.wikipedia.org/wiki/Gaussian_blur
void generate_gaussian_kernel(boost::gil::gray32f_view_t dst, float sigma) {
    for (boost::gil::gray32f_view_t::coord_t y = 0; y < dst.height(); ++y)
    {
        for (boost::gil::gray32f_view_t::coord_t x = 0; x < dst.height(); ++x)
        {
            const float power = -(x * x +  y * y) / (2 * sigma);
            const float nominator = std::exp(power);
            const float value = nominator / (2 * boost::gil::pi * sigma * sigma);
            dst(x, y).at(std::integral_constant<int, 0>{}) = value;
        }
    }
}

}} // namespace boost::gil

#endif
