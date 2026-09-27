#pragma once

#include <filesystem>

#include <opencv2/core.hpp>

#include <sdv/camera.h>

namespace sdv {

// YAML camera description (EUCM; pinhole is alpha 0, beta 1):
//   width, height, fx, fy, cx, cy, alpha, beta   (alpha/beta default to the pinhole values)
//   mask: optional 1-channel image path, relative to the YAML file; pixels above 128 are valid.
struct CameraConfig {
  Camera camera;  // without mask
  cv::Mat mask;   // CV_8UC1 at camera resolution, or empty
};

CameraConfig loadCameraConfig(const std::filesystem::path& file);

// Input images are resized by `scale` and then cropped at the right and bottom to a multiple of `sizeMultiple`
// (the pyramid needs it; cropping there keeps the principal point). Returns the matching camera with its
// validity mask attached (if the config has one).
Camera prepareCamera(const CameraConfig& config, double scale, int sizeMultiple);
cv::Mat prepareImage(const cv::Mat& image, double scale, const Camera& prepared);

}  // namespace sdv
