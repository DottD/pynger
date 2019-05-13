#ifndef AdaptiveThreshold_hpp
#define AdaptiveThreshold_hpp

#include <vector>
#include <functional>
#include <opencv2/opencv.hpp>

namespace fm {
    /* Function that returns the adaptive threshold for the input image */
    float adaptiveThreshold(const cv::Mat1f& image,
                            const float& hint = FLT_MIN);
    
    /* Function that produces a binary image */
    void medianAdaptiveThreshold(cv::Mat1b& bin,
                                 const cv::Mat1f& image,
                                 const cv::Mat1b& mask,
                                 const int& binarizationStep,
                                 const float& percentile);
}

#endif /* AdaptiveThreshold_hpp */
