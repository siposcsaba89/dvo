#pragma once

#include <vector>

#include <Eigen/Core>
#include <sophus/se3.hpp>

namespace sdv {

// dst = scale * R * src + t
struct SimilarityTransform {
  double scale = 1.0;
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();

  Eigen::Vector3d apply(const Eigen::Vector3d& p) const { return scale * R * p + t; }
  Sophus::SE3d applyToPose(const Sophus::SE3d& T_w_c) const;
};

// Umeyama 1991.
SimilarityTransform alignPoints(const std::vector<Eigen::Vector3d>& src,
                                const std::vector<Eigen::Vector3d>& dst, bool estimateScale);

struct AteResult {
  SimilarityTransform alignment;
  double rmse = 0, mean = 0, median = 0, max = 0;
};

AteResult absoluteTrajectoryError(const std::vector<Sophus::SE3d>& gt,
                                  const std::vector<Sophus::SE3d>& est, bool estimateScale);

struct SegmentErrorResult {
  double translationPercent = 0;
  double rotationDegPer100m = 0;
  int numSegments = 0;
};

// KITTI-benchmark style drift over 100..800 m segments; `est` must be aligned to gt.
SegmentErrorResult segmentDriftError(const std::vector<Sophus::SE3d>& gt,
                                     const std::vector<Sophus::SE3d>& est);


struct RelativePoseErrorResult {
  double translationRmse = 0;  // gt units
  double rotationRmseDeg = 0;
  int numPairs = 0;
};

// Relative pose error between poses `delta` apart (Sturm et al., IROS 2012); `est` translations are multiplied by
// `scale` first (Sim3 scale of a monocular estimate).
RelativePoseErrorResult relativePoseError(const std::vector<Sophus::SE3d>& gt, const std::vector<Sophus::SE3d>& est,
                                          size_t delta = 1, double scale = 1.0);

}  // namespace sdv
