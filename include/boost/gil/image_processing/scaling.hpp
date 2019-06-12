#include <boost/gil/image_view.hpp>
#include <boost/gil/rgb.hpp>
#include <boost/gil/pixel.hpp>
#include <boost/gil/image_processing/numeric.hpp>

namespace boost{ namespace gil{
/// \defgroup ScalingAlgorithms
/// \brief Algorthims suitable for rescaling
///
/// These algorithms are used to improve image
/// quality after image resizing is made.

/// \defgroup DownScalingAlgorithms
/// \ingroup ScalingAlgorithms
/// \brief Algorthims suitable for downscaling
///
/// These algorithms provide best results when used
/// for downscaling. Using for upscaling will probably
/// provide less than good results.


/// \brief a single step of lanczos downscaling
/// \ingroup DownScalingAlgorithms
///
/// Use this algorithm to scale down source image
/// into a smaller image with reasonable quality.
/// Do note that having a look at the output once
/// is a good idea, since it might have ringing
/// artifacts.
template <typename ImageView>
void lanczos_at(
    std::ptrdiff_t source_x, 
    std::ptrdiff_t source_y, 
    std::ptrdiff_t target_x, 
    std::ptrdiff_t target_y, 
    ImageView input_view, 
    ImageView output_view, 
    std::ptrdiff_t a) 
{
    using pixel_t = typename std::remove_reference<
                      decltype(std::declval<ImageView>()(0, 0))
                    >::type;
    // C++11 doesn't allow auto in lambdas
    using channel_t = typename std::remove_reference<
                        decltype(
                            std::declval<pixel_t>().at(
                                std::integral_constant<int, 0>{}
                            )
                        )
                       >::type;
    pixel_t result_pixel;
    boost::gil::static_transform(result_pixel, result_pixel, 
        [](channel_t) { return static_cast<channel_t>(0); });
    
    for (std::ptrdiff_t y_i = std::max(source_y - a + 1l, 0l); 
         y_i <= std::min(source_y + a, input_view.height() - 1l);
         ++y_i) 
        {
        for (std::ptrdiff_t x_i = std::max(source_x - a + 1l, 0l); 
             x_i <= std::min(source_x + a, input_view.width() - 1l); 
             ++x_i) 
        {
            double lanczos_response = boost::gil::lanczos(source_x - x_i, a) 
                                      * boost::gil::lanczos(source_y - y_i, a);
            auto op = [lanczos_response](channel_t prev, channel_t next) 
            {
                return static_cast<channel_t>(prev + next * lanczos_response);
            };
            boost::gil::static_transform(result_pixel, 
                                         input_view(source_x, source_y), 
                                         result_pixel, 
                                         op);
        }
    }

    output_view(target_x, target_y) = result_pixel;
}

/// \brief Complete Lanczos algorithm
/// \ingroup DownScalingAlgorithms
///
/// This algorithm does full pass over
/// resulting image and convolves pixels from
/// original image. Do note that it might be a good
/// idea to have a look at test output as there
/// might be ringing artifacts.
/// Based on wikipedia article:
/// https://en.wikipedia.org/wiki/Lanczos_resampling
/// with standardinzed cardinal sin (sinc)
template <typename ImageView>
void scale_lanczos(ImageView input_view, ImageView output_view, std::ptrdiff_t a) 
{
    double scale_x = (static_cast<double>(output_view.width())) 
                     / static_cast<double>(input_view.width());
    double scale_y = (static_cast<double>(output_view.height())) 
                     / static_cast<double>(input_view.height());

    for (std::ptrdiff_t y = 0; y < output_view.height(); ++y) 
    {
        for (std::ptrdiff_t x = 0; x < output_view.width(); ++x) 
        {
            boost::gil::lanczos_at(
                x / scale_x, 
                y / scale_y, 
                x, 
                y, 
                input_view, 
                output_view, 
                a);
        }
    }
}
}}
