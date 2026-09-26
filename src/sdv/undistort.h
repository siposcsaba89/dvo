#pragma once

#include <opencv2/core.hpp>

#include <sdv/camera.h>

namespace sdv {

// Pinhole camera of the same image size, centred, with focal length focalScale * cam.fx (EUCM matches a pinhole
// with the same focal length at the image centre; focalScale < 1 keeps more of a wide field of view).
Camera virtualPinhole(const Camera& cam, double focalScale = 1.0);

struct UndistortMap {
  cv::Mat mapX, mapY;  // CV_32F, dst pixel -> src pixel (-1 where dst has no source)
};

UndistortMap makeUndistortMap(const Camera& src, const Camera& dst);
cv::Mat undistort(const cv::Mat& image, const UndistortMap& map);

}  // namespace sdv
