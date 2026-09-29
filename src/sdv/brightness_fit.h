#pragma once

#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <sophus/se3.hpp>

#include <sdv/camera.h>
#include <sdv/photometric.h>

namespace sdv {

struct BrightnessFitSettings {
  double huberThreshold = 9.0;
  double gradientWeightC = 50.0;
  int hostWindow = 5;  // points hosted within this many keyframes; nearby hosts are rarely occluded
  int minPoints = 50;
  int iterations = 10;
};

// A world point with its irradiance e^-a_h (I_h - b_h), i.e. in the brightness gauge of the cameras it came from.
struct IrradiancePoint {
  Eigen::Vector3d position;
  float irradiance;
  int keyframe;  // host
};

struct BrightnessFitResult {
  std::vector<AffineBrightness> affine;  // per keyframe
  std::vector<int> inliers;  // per keyframe; 0 where too few points took the nearest fitted keyframe
};

// Affine brightness per keyframe of a camera that was not adjusted, with poses and points fixed: I = e^a J + b
// (the residual model of the odometry) fitted to the irradiance J of the points, Huber-weighted against
// occlusions and misalignment. images: the camera's keyframe images (grey or BGR, prepared for `cam`).
BrightnessFitResult fitCameraBrightness(const Camera& cam, const std::vector<Sophus::SE3d>& T_c_w,
                                        const std::vector<cv::Mat>& images, const std::vector<IrradiancePoint>& points,
                                        const BrightnessFitSettings& settings = {});

}  // namespace sdv
