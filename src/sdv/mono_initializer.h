#pragma once

#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <sophus/se3.hpp>

#include <sdv/camera.h>
#include <sdv/image_pyramid.h>
#include <sdv/photometric.h>

namespace sdv {

struct MonoInitSettings {
  int pointsLevel0 = 2000;
  double levelPointFactor = 0.5;
  PhotometricSettings photometric;
  double outlierCutoff = 40.0;
  double smoothWeight = 50.0;
  double gaugeWeight = 1.0;  // pulls rho toward its mean at reset, fixing the monocular scale
  int numNeighbours = 6;
  std::vector<int> iterations = {8, 8, 10, 10, 12};  // index = pyramid level
  double minTranslationFlow = 6.0;                  // median pixels at level 0
  int minFrames = 2;
  double minInlierRatio = 0.5;
  double minVisibleRatio = 0.5;
  std::vector<double> firstFrameTranslations = {0.02, 0.05, 0.1};  // divided by mean rho, along +-x, +-y, +-z
  int firstFrameCandidates = 2;
};

struct InitPoint {
  Eigen::Vector2d uv;
  Eigen::Vector3d bearing;
  double rho;
  double relativeSigma;  // 1 / (rho * sqrt(photometric Hessian)), unit intensity noise
};

struct MonoInitResult {
  bool initialized = false;
  bool reset = false;
  Sophus::SE3d T_t_h;
  AffineBrightness affine;
  double translationFlow = 0;
  double inlierRatio = 0;
  double visibleRatio = 0;
  double rmse = 0;
  int numFrames = 0;
};

// Direct two-frame initialisation: first-frame points start at rho = 1 (or a prior) and are optimised jointly
// with the pose of each new frame (Schur complement on the pose) until there is enough translation.
// Scale is normalised to mean rho = 1.
class MonoInitializer {
 public:
  MonoInitializer(const Camera& cam, int numLevels, MonoInitSettings settings = {});

  // rhoPrior: optional CV_32F level-0 inverse distance (<= 0 = unknown) instead of the constant start.
  void reset(const ImagePyramid& first, const cv::Mat& rhoPrior = {});
  MonoInitResult addFrame(const ImagePyramid& frame);

  std::vector<InitPoint> points() const;
  int numPoints(int level) const { return static_cast<int>(m_levels[level].points.size()); }

 private:
  struct Point {
    PatternPoint pattern;
    Eigen::Vector3d bearing;
    double rho = 1.0;
    double rhoStart = 1.0;
    double hPhoto = 0.0;
    double energy = 0.0;
    bool valid = false;
    int parent = -1;
    std::vector<int> neighbours;
  };
  struct Level {
    Camera cam;
    std::vector<Point> points;
  };
  struct State {
    Sophus::SE3d T_t_h;
    AffineBrightness affine;
  };
  struct System {
    Eigen::Matrix<double, 8, 8> Hxx;
    Eigen::Matrix<double, 8, 1> gx;
    std::vector<Eigen::Matrix<double, 8, 1>> Hxr;
    std::vector<double> Hrr, gr, hPhoto;
    std::vector<double> pointEnergy, pointPrior;
    std::vector<char> valid;
    double energy = 0;
    int numResiduals = 0;  // of points in view
    int numInliers = 0;
    double photometricEnergy = 0;
    double visibleEnergy = 0;
    int numVisible = 0;

    double inlierRatio() const { return numResiduals > 0 ? double(numInliers) / numResiduals : 0.0; }
    // Compares different motion hypotheses; total energy would favour motions that keep points in view.
    double visibleMeanEnergy() const { return numVisible > 0 ? visibleEnergy / numVisible : 1e30; }
  };

  // Points leaving the view keep their fallback energy so that leaving is neither rewarded nor penalised.
  System linearize(int level, const ImageLevel& img, const State& state, const std::vector<double>& rho,
                   const std::vector<double>& fallbackEnergy, bool withJacobians) const;
  System optimizeLevel(int level, const ImageLevel& img, State& state, int maxIterations);
  System optimizePyramid(const ImagePyramid& frame, State& state);
  // Ranks start poses by the energy reached on the top level; returns the best `keep` starts.
  std::vector<State> prescreen(const ImagePyramid& frame, const std::vector<State>& starts, int keep);
  // Rotation + affine only, points at infinity: a start that does not depend on the unknown depths.
  void alignRotation(const ImagePyramid& frame, State& state) const;
  double translationFlow(const State& state) const;
  void normalizeScale();

  Camera m_camera;
  int m_numLevels;
  MonoInitSettings m_settings;
  std::vector<Level> m_levels;
  State m_state, m_prevState;
  int m_numFrames = 0;
  double m_gaugeRho = 1.0;
};

}  // namespace sdv
