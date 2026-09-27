#pragma once

#include <utility>
#include <vector>

#include <Eigen/Core>
#include <sophus/se3.hpp>

#include <sdv/keyframe_features.h>
#include <sdv/pose_graph.h>
#include <sdv/rig.h>

namespace sdv {

struct GlobalBASettings {
  int sequentialNeighbours = 3;  // each keyframe is matched with this many following keyframes
  double ratio = 0.8;
  int maxHamming = 64;
  double gatePixels = 5.0;  // matches must agree with the initial poses this well
  double huberPixels = 1.5;
  double relativeDepthSigma = 0.05;  // prior on the map depth of a feature (0 = off)
  // Odometry edges between consecutive keyframes, sigmas of the pose graph times this factor (0 = off).
  double odometrySigmaFactor = 5.0;
  PoseGraphSettings odometry;
  double outlierPixels = 3.0;
  int rounds = 2;  // solves, observations above outlierPixels removed in between
  int iterations = 50;
};

struct GlobalBAResult {
  std::vector<Sophus::SE3d> T_w_b;
  std::vector<Eigen::Vector3d> points;  // world, one per track
  size_t observations = 0, loopObservations = 0;  // loopObservations: tracks spanning a loop pair
  double rmseBefore = 0, rmseAfter = 0;  // pixels, over the kept observations
  int iterations = 0;
};

// Feature bundle adjustment over all keyframes of a run: ORB matches between neighbouring keyframes and loop pairs
// become tracks; keyframe poses (rig extrinsics fixed, first keyframe fixed) and track points are optimised on the
// angular reprojection error (in pixels of the observing camera), with priors from the map depth of features and
// weak odometry edges. initial_T_w_b: e.g. the pose graph result.
GlobalBAResult bundleAdjustKeyframes(const Rig& rig, const std::vector<KeyframeRecord>& records,
                                     const std::vector<Sophus::SE3d>& initial_T_w_b,
                                     const std::vector<std::pair<int, int>>& loopPairs,
                                     const GlobalBASettings& settings = {});

}  // namespace sdv
