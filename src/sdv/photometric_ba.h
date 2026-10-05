#pragma once

#include <array>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <sophus/se3.hpp>

#include <sdv/keyframe_features.h>
#include <sdv/photometric.h>
#include <sdv/pose_graph.h>
#include <sdv/rig.h>

namespace sdv {

struct PhotometricBASettings {
  int maxPointsPerImage = 0;  // evenly subsample the hosted points of each keyframe image (0 = all)
  int neighbours = 5;  // each point is observed in the keyframes this many before and after its host
  int loopNeighbours = 2;  // ... and in these around the keyframes its host is linked to by a loop
  // ... and in the closest keyframes within this distance of its host that are not temporal neighbours (revisits;
  // the pose graph has aligned them already): up to spatialTargets of them.
  double spatialRadius = 6.0;
  int spatialTargets = 8;
  int maxCrossTargets = 8;  // loop and revisit target keyframes per point, the closest to its host (0 = all); memory
  double crossMaxInitialPixelError = 30.0;  // initial check for these revisit residuals
  double maxViewAngleDeg = 40.0;  // host and target rays of a point
  double maxInitialPixelError = 20.0;  // initial photometric error per pattern pixel for a residual to be used
  int minInitialResiduals = 3;  // a point takes part only with this many residuals that pass the initial check
  double huber = 9.0;  // per pattern pixel, as in the odometry
  double outlierPixelError = 12.0;  // after the first round
  double odometrySigmaFactor = 10.0;  // weak odometry edges (0 = off)
  PoseGraphSettings odometry;
  int rounds = 2;
  // > 0: bounded memory by block-coordinate descent over blocks of this many consecutive keyframes (their poses,
  // brightness and hosted inverse depths; the rest fixed), blockSweeps sweeps per round with borders shifted by half
  // a block. 0: one joint problem.
  int blockKeyframes = 0;
  int blockSweeps = 2;
  size_t blockCoarseResiduals = 3000000;
  // Linear solver: sparse Schur with Eigen's (single-threaded) Cholesky, AMD or nested-dissection ordering; or
  // iterative Schur (conjugate gradients, Schur-Jacobi preconditioner, multi-threaded, for large problems).
  enum class Solver { SparseAmd, SparseNesdis, Iterative };
  Solver solver = Solver::SparseAmd;  // before the blocks: one joint solve with every m-th point within this
  // Custom: our Levenberg-Marquardt (photometric_ba_solver.h; no per-residual storage). Ceres: the reference, and always
  // with refineExtrinsics / refineIntrinsics; `solver` applies to it only.
  enum class Optimizer { Custom, Ceres };
  Optimizer optimizer = Optimizer::Custom;
  int iterations = 30;
  PhotometricSettings photometric;
  // Optional: also refine the rig extrinsics T_c_b (one per camera, shared by all keyframes; camera 0 stays fixed as
  // the body reference), with a prior towards the input rig. Static residuals between overlapping cameras fix the
  // relative extrinsics; temporal ones also the camera orientation relative to the body motion.
  bool refineExtrinsics = false;
  double extrinsicSigmaT = 0.05;     // m
  double extrinsicSigmaRDeg = 2.0;
  // With refineExtrinsics: hold the sum of the distances between all camera centres (the metric scale of the rig) at
  // its input value, so that the translations can move freely otherwise (with extrinsicSigmaT as a weak prior).
  bool extrinsicFixScale = false;
  // Optional: also refine the intrinsics (fx, fy, cx, cy, alpha, beta) of these cameras (empty = all), with a prior
  // towards the rig cameras; the extrinsics enter as with refineExtrinsics (held fixed without it). Fixed camera
  // translations keep the metric scale.
  bool refineIntrinsics = false;
  std::vector<int> intrinsicCameras;
  double intrinsicSigmaFocal = 0.01;  // relative
  double intrinsicSigmaCenter = 2.0;  // px
  double intrinsicSigmaAlpha = 0.02;
  double intrinsicSigmaBeta = 0.05;
};

// Photometric rmse per pattern pixel of the residuals from camera `host` into camera `target`, over the residuals
// kept after the adjustment, with the initial and the final estimate (the per-camera consistency check).
struct CameraPairError {
  int host, target;
  bool sameKeyframe;  // static residuals (other camera of the host keyframe) or temporal ones
  size_t residuals;
  double rmseBefore, rmseAfter;
};

struct PhotometricBAResult {
  std::vector<Sophus::SE3d> T_w_b;
  std::vector<std::vector<AffineBrightness>> affine;
  std::vector<Eigen::Vector3d> points;  // world, all hosted points of the records that were used
  std::vector<double> pointDistance;  // from the host camera
  std::vector<std::array<int, 2>> pointHost;  // keyframe (record index), camera
  std::vector<Eigen::Vector2d> pointUv;  // in the host camera image
  std::vector<int> pointResiduals;  // residuals left after outlier removal (0: not used)
  // Inverse-depth standard deviation relative to the inverse depth, for unit photometric noise, from the depth
  // information of the final residuals (other parameters fixed), as MapPoint::relativeDepthSigma of active points.
  std::vector<double> pointDepthSigma;
  size_t residuals = 0, loopResiduals = 0;
  double rmseBefore = 0, rmseAfter = 0;  // intensity per pattern pixel over the used residuals
  int iterations = 0;
  std::vector<Sophus::SE3d> T_c_b;  // rig extrinsics after the adjustment (the input ones without refineExtrinsics)
  std::vector<Camera> cameras;  // rig cameras after the adjustment (the input ones without refineIntrinsics)
  std::vector<CameraPairError> cameraPairs;
};

// Global photometric bundle adjustment over all keyframes: every map point stored in the records (host keyframe
// camera, pixel, inverse distance) gets pattern residuals in the cameras of nearby keyframes and of keyframes linked
// by loops; keyframe poses, affine brightness per keyframe camera and inverse depths are optimised (rig extrinsics
// fixed unless settings.refineExtrinsics; first keyframe pose and its camera-0 brightness fixed). images[k][c]: grey keyframe images at the scale of
// the records' cameras. Needs initial poses within ~1-2 px, e.g. after the pose graph.
PhotometricBAResult photometricBundleAdjust(const Rig& rig, const std::vector<KeyframeRecord>& records,
                                            const std::vector<std::vector<cv::Mat>>& images,
                                            const std::vector<Sophus::SE3d>& initial_T_w_b,
                                            const std::vector<std::pair<int, int>>& loopPairs,
                                            const PhotometricBASettings& settings = {});

}  // namespace sdv
