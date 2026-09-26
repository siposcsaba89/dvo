#pragma once

#include <array>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <sophus/se3.hpp>

#include <sdv/camera.h>
#include <sdv/image_pyramid.h>
#include <sdv/photometric.h>

namespace sdv {

struct FrameParams {
  Sophus::SE3d T_c_w;
  AffineBrightness affine;
  double exposure = 1.0;
};

struct WindowPixelResidual {
  double r = 0;
  double weight = 0;  // gradient weight * Huber IRLS weight
  Eigen::Matrix<double, 1, 8> dHost, dTarget;  // (xi, a, b) with left increments on T_c_w
  double dRho = 0;
};

struct WindowPatternResidual {
  std::array<WindowPixelResidual, kPatternSize> pixels;
  double energy = 0;
};

// Residual at the current parameters; frame Jacobians at the linearisation parameters host0/target0 (FEJ),
// image gradients at the current projection. DSO §2.3.
bool evaluateWindowResidual(const PatternPoint& point, double rho, const FrameParams& host,
                            const FrameParams& target, const FrameParams& host0, const FrameParams& target0,
                            const Camera& cam, const ImageLevel& targetImg, const PhotometricSettings& settings,
                            WindowPatternResidual& out);

struct WindowSettings {
  PhotometricSettings photometric;
  double outlierThreshold = 20.0;  // residual is an outlier if its energy exceeds that of |r| = threshold per pixel
  double affinePriorA = 1e3;
  double affinePriorB = 1.0;
  double firstFramePrior = 1e10;
  int maxIterations = 6;
  double minRhoHessian = 1e-3;  // points with less depth information are dropped instead of marginalised
};

enum class ResidualState { Good, OutOfBounds, Outlier };

struct WindowResidual {
  int target;  // frame id
  ResidualState state = ResidualState::Good;
  double energy = 0;
};

struct WindowPoint {
  int id;
  int host;  // frame id
  PatternPoint pattern;
  Eigen::Vector3d bearing;
  double rho;
  double hRho = 0;  // photometric depth information from the last optimisation
  std::vector<WindowResidual> residuals;

  int numGood() const;
};

struct WindowFrame {
  int id;
  std::shared_ptr<const ImagePyramid> image;
  FrameParams params;
  FrameParams linearization;  // fixed once the frame is part of the marginalisation prior
  bool inPrior = false;

  Eigen::Matrix<double, 8, 1> delta() const;
};

struct WindowOptimizationResult {
  int iterations = 0;
  double initialEnergy = 0, finalEnergy = 0;
  int numGood = 0, numOutliers = 0, numOutOfBounds = 0;
};

// Sliding-window photometric bundle adjustment over keyframe poses, affine brightness and point inverse
// distances, with Schur-complement marginalisation of old frames (DSO §2.3).
class WindowOptimizer {
 public:
  WindowOptimizer(const Camera& cam, WindowSettings settings = {});

  int addFrame(std::shared_ptr<const ImagePyramid> image, const Sophus::SE3d& T_c_w,
               const AffineBrightness& affine = {}, double exposure = 1.0);
  // Returns the point id, or -1 if the pattern does not fit into the host image.
  int addPoint(int hostId, const Eigen::Vector2d& uv, double rho);
  void removePoint(int pointId);
  void setFrameParams(int frameId, const FrameParams& params) { m_frames[frameIndex(frameId)].params = params; }

  WindowOptimizationResult optimize(int maxIterations = -1);
  void marginalizeFrame(int frameId);

  const std::vector<WindowFrame>& frames() const { return m_frames; }
  const std::vector<WindowPoint>& points() const { return m_points; }
  const WindowFrame& frame(int frameId) const { return m_frames[frameIndex(frameId)]; }
  const Camera& camera() const { return m_camera; }

 private:
  struct PointBlock {
    double Hrr = 0, gr = 0;
    std::vector<Eigen::Matrix<double, 8, 1>> Hfr;  // per frame index
  };
  struct System {
    Eigen::MatrixXd H;
    Eigen::VectorXd g;
    std::vector<PointBlock> points;
  };

  int frameIndex(int frameId) const;
  // Accumulates the photometric system of the given points (all if empty) over Good residuals.
  System linearize(const std::vector<size_t>& pointIndices) const;
  void classifyResiduals();
  double energy() const;
  double priorEnergy() const;
  void addPriors(System& sys) const;

  Camera m_camera;
  WindowSettings m_settings;
  std::vector<WindowFrame> m_frames;
  std::vector<WindowPoint> m_points;
  Eigen::MatrixXd m_priorH;  // over m_frames order, in delta coordinates
  Eigen::VectorXd m_priorB;
  int m_nextFrameId = 0;
  int m_nextPointId = 0;
};

}  // namespace sdv
