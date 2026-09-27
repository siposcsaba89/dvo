#pragma once

#include <vector>

#include <sophus/se3.hpp>

#include <sdv/loop_detector.h>

namespace ceres {
class CostFunction;
}

namespace sdv {

// Relative pose residual between two T_w_b parameter blocks (Sophus SE3 layout), weighted by the sigmas.
ceres::CostFunction* relativePoseCost(const Sophus::SE3d& T_a_b, double sigmaT, double sigmaRDeg);

struct PoseGraphSettings {
  // Odometry edge between consecutive keyframes: sigma = base + rate * distance between them.
  double odometryTranslationBase = 0.005, odometryTranslationRate = 0.01;  // m, m per m
  double odometryRotationBaseDeg = 0.02, odometryRotationRateDegPerMeter = 0.005;
  double loopTranslationSigma = 0.05, loopRotationSigmaDeg = 0.2;
  // Loops arrive verified, so they enter without a robust loss (which could not close a loop far outside its
  // sigma); after each solve the worst loop whose normalised residual exceeds this is removed and the graph solved
  // again.
  double loopRejectSigma = 5.0;
  double loopRejectMedianFactor = 3.0;  // ... and exceeds this times the median loop residual
  int maxRejectRounds = 20;
  int iterations = 100;
};

struct PoseGraphResult {
  std::vector<Sophus::SE3d> T_w_b;  // optimised keyframe poses
  double initialCost = 0, finalCost = 0;
  int iterations = 0;
  std::vector<char> loopAccepted;  // per input loop
  int loopsRejected = 0;
};

// SE3 pose graph over keyframes (metric: stereo or rig): consecutive odometry edges plus loop edges; the first
// keyframe is fixed. Residual log(T_a_b_measured^-1 * T_w_a^-1 * T_w_b), weighted by the edge sigmas. Loops that
// contradict the others are removed one at a time (see loopRejectSigma).
PoseGraphResult optimizePoseGraph(const std::vector<Sophus::SE3d>& odometry_T_w_b,
                                  const std::vector<LoopConstraint>& loops, const PoseGraphSettings& settings = {});

// World-frame correction of every input frame after the pose graph (T_w_b' = C * T_w_b): keyframes get theirs
// exactly, frames in between interpolate between the neighbouring keyframes.
class PoseCorrection {
 public:
  PoseCorrection() = default;
  PoseCorrection(const std::vector<int>& keyframeFrames, const std::vector<Sophus::SE3d>& before,
                 const std::vector<Sophus::SE3d>& after);

  Sophus::SE3d at(int frameIndex) const;
  bool empty() const { return m_frames.empty(); }

 private:
  std::vector<int> m_frames;
  std::vector<Sophus::SE3d> m_corrections;
};

}  // namespace sdv
