#pragma once

#include <vector>

#include <Eigen/Core>
#include <sophus/se3.hpp>

#include <sdv/camera.h>
#include <sdv/image_pyramid.h>
#include <sdv/photometric.h>
#include <sdv/rig.h>

namespace sdv {

struct TrackingSettings {
  double huberThreshold = 9.0;
  double gradientWeightC = 50.0;
  double outlierCutoff = 40.0;
  double minInlierRatio = 0.5;
  double minVisibleRatio = 0.2;
  int maxCutoffIncreases = 3;
  std::vector<int> maxIterations = {6, 8, 10, 15, 20};  // index = pyramid level
  int hypothesisIterations = 5;
  double convergenceEps = 1e-5;
  double border = 2.0;
  double affinePriorA = 0.0;
  double affinePriorB = 0.0;
  // Also refine the first hypothesis (the motion prediction) from this pyramid level down, and keep it if its
  // finest-level energy is lower than that of a coarse-to-fine result that jumped towards the reference (-1 = off).
  int predictionLevel = -1;
  double predictionJump = 0.5;  // minimum jump, relative to the predicted step from the previous frame
};

class ReferenceFrame {
 public:
  struct Point {
    Eigen::Vector3d bearing;
    Eigen::Vector2d uv;
    double rho;
    float intensity;
    float gradientWeight;
  };

  // Level-0 pixel positions with inverse distance; coarser levels average 2x2 blocks.
  ReferenceFrame(const Camera& cam, const ImagePyramid& pyramid, const std::vector<Eigen::Vector2i>& pixels,
                 const std::vector<double>& rho, const AffineBrightness& affine, double gradientWeightC);

  int numLevels() const { return static_cast<int>(m_points.size()); }
  const std::vector<Point>& points(int level) const { return m_points[level]; }
  const Camera& camera(int level) const { return m_cameras[level]; }
  const AffineBrightness& affine() const { return m_affine; }

 private:
  std::vector<std::vector<Point>> m_points;
  std::vector<Camera> m_cameras;
  AffineBrightness m_affine;
};

struct TrackingResult {
  bool ok = false;
  Sophus::SE3d T_t_h;  // body motion from the reference keyframe
  std::vector<AffineBrightness> affine;  // per camera
  double rmse = 0;
  double inlierRatio = 0;
  double visibleRatio = 0;
  double meanFlow = 0;
  int hypothesis = -1;
};

// Direct alignment of a new rig frame against the reference keyframe: body motion and per-camera brightness.
class FrameTracker {
 public:
  explicit FrameTracker(TrackingSettings settings) : m_settings(std::move(settings)) {}

  // One reference and one target pyramid per rig camera; a point is aligned within its own camera. T_prev_ref, the
  // previous frame w.r.t. the reference, gives the predicted step for TrackingSettings::predictionLevel.
  TrackingResult track(const Rig& rig, const std::vector<ReferenceFrame>& refs,
                       const std::vector<const ImagePyramid*>& targets, const std::vector<Sophus::SE3d>& hypotheses,
                       const std::vector<AffineBrightness>& initialAffine,
                       const Sophus::SE3d* T_prev_ref = nullptr) const;
  TrackingResult track(const ReferenceFrame& ref, const ImagePyramid& target,
                       const std::vector<Sophus::SE3d>& hypotheses, const AffineBrightness& initialAffine) const;

 private:
  struct State {
    Sophus::SE3d T_t_h;
    std::vector<AffineBrightness> affine;
  };
  struct System {
    Eigen::MatrixXd H;
    Eigen::VectorXd g;
    double energy = 0;
    int numInliers = 0;
    int numVisible = 0;
    int numPoints = 0;

    double normalizedEnergy() const { return numVisible > 0 ? energy / numVisible : 1e30; }
    double inlierRatio() const { return numVisible > 0 ? double(numInliers) / numVisible : 0.0; }
  };
  struct CameraSystem {
    Eigen::Matrix<double, 8, 8> H;  // camera-frame pose increment, then (a, b)
    Eigen::Matrix<double, 8, 1> g;
    double energy = 0;
    int numInliers = 0, numVisible = 0, numPoints = 0;
  };

  CameraSystem linearizeCamera(const ReferenceFrame& ref, const ImageLevel& img, int level,
                               const Sophus::SE3d& T_t_h, const AffineBrightness& affine, double cutoff,
                               bool withJacobians) const;
  System linearize(const Rig& rig, const std::vector<ReferenceFrame>& refs,
                   const std::vector<const ImagePyramid*>& targets, int level, const State& state, double cutoff,
                   bool withJacobians) const;
  System optimizeLevel(const Rig& rig, const std::vector<ReferenceFrame>& refs,
                       const std::vector<const ImagePyramid*>& targets, int level, State& state, double cutoff,
                       int maxIterations) const;
  double meanFlow(const Rig& rig, const std::vector<ReferenceFrame>& refs, const State& state) const;

  TrackingSettings m_settings;
};

// T_prev_ref: previous frame w.r.t. reference; T_prev_prevprev: last inter-frame motion.
std::vector<Sophus::SE3d> makeMotionHypotheses(const Sophus::SE3d& T_prev_ref, const Sophus::SE3d& T_prev_prevprev);

}  // namespace sdv
