#pragma once

#include <filesystem>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <sophus/se3.hpp>

#include <sdv/camera.h>
#include <sdv/image_pyramid.h>
#include <sdv/rig.h>

namespace sdv {

struct FeatureSettings {
  int featuresPerImage = 1000;
  int gridColumns = 8, gridRows = 5;  // keypoints are spread over this grid
  float fastThreshold = 12.f;
  double depthRadius = 3.0;  // pixels: map points this close to a keypoint give its depth
  double depthConsistency = 0.05;  // relative inverse-depth spread allowed among them
};

// ORB keypoints of one camera image of a keyframe, with depth where the odometry map supports it.
struct CameraFeatures {
  std::vector<cv::KeyPoint> keypoints;
  std::vector<Eigen::Vector3f> bearings;
  std::vector<float> rho;  // inverse distance from the camera; 0 = unknown
  cv::Mat descriptors;  // CV_8U, one row per keypoint

  size_t size() const { return keypoints.size(); }
  int numWithDepth() const;
};

struct KeyframeRecord {
  int frameIndex;
  Sophus::SE3d T_w_b;
  std::vector<CameraFeatures> cameras;
};

cv::Mat toGray8(const ImageLevel& img);
CameraFeatures extractFeatures(const Camera& cam, const cv::Mat& gray8, const FeatureSettings& settings);
// uv, rho: map points projected into this camera (pixel, inverse distance).
void assignDepth(CameraFeatures& features, const std::vector<Eigen::Vector2f>& uv, const std::vector<float>& rho,
                 const FeatureSettings& settings);

void saveKeyframeRecords(const std::filesystem::path& file, const Rig& rig, const std::vector<KeyframeRecord>& records);
std::vector<KeyframeRecord> loadKeyframeRecords(const std::filesystem::path& file, Rig* rig = nullptr);

}  // namespace sdv
