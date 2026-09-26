#pragma once

#include <vector>

#include <Eigen/Core>
#include <sophus/se3.hpp>

#include <sdv/camera.h>
#include <sdv/image_pyramid.h>
#include <sdv/photometric.h>

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
  Sophus::SE3d T_t_h;
  AffineBrightness affine;
  double rmse = 0;
  double inlierRatio = 0;
  double visibleRatio = 0;
  double meanFlow = 0;
  int hypothesis = -1;
};

class FrameTracker {
 public:
  explicit FrameTracker(TrackingSettings settings) : m_settings(std::move(settings)) {}

  TrackingResult track(const ReferenceFrame& ref, const ImagePyramid& target,
                       const std::vector<Sophus::SE3d>& hypotheses, const AffineBrightness& initialAffine,
                       double exposureRatio = 1.0) const;

 private:
  struct State {
    Sophus::SE3d T_t_h;
    AffineBrightness affine;
  };
  struct System {
    Eigen::Matrix<double, 8, 8> H;
    Eigen::Matrix<double, 8, 1> g;
    double energy = 0;
    int numInliers = 0;
    int numVisible = 0;
    int numPoints = 0;

    double normalizedEnergy() const { return numVisible > 0 ? energy / numVisible : 1e30; }
    double inlierRatio() const { return numVisible > 0 ? double(numInliers) / numVisible : 0.0; }
  };

  System linearize(const ReferenceFrame& ref, const ImageLevel& img, int level, const State& state,
                   double cutoff, double exposureRatio, bool withJacobians) const;
  System optimizeLevel(const ReferenceFrame& ref, const ImageLevel& img, int level, State& state,
                       double cutoff, double exposureRatio, int maxIterations) const;
  double meanFlow(const ReferenceFrame& ref, const State& state) const;

  TrackingSettings m_settings;
};

// T_prev_ref: previous frame w.r.t. reference; T_prev_prevprev: last inter-frame motion.
std::vector<Sophus::SE3d> makeMotionHypotheses(const Sophus::SE3d& T_prev_ref, const Sophus::SE3d& T_prev_prevprev);

}  // namespace sdv
