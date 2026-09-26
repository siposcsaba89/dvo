#pragma once

#include <opencv2/core.hpp>

#include <sdv/camera.h>

namespace sdv {

// Rectified pinhole stereo only; used to validate components before monocular depth exists.
// Returns CV_32F inverse distance along the unit bearing, 0 where invalid.
cv::Mat stereoInverseDistance(const cv::Mat& left, const cv::Mat& right, const Camera& cam, double baseline,
                              double minDisparity = 2.0);

}  // namespace sdv
