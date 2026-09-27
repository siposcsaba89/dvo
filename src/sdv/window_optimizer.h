#pragma once

#include <array>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <sophus/se3.hpp>

#include <sdv/camera.h>
#include <sdv/image_pyramid.h>
#include <sdv/photometric.h>
#include <sdv/rig.h>

namespace sdv {

// Pose and brightness of one camera.
struct FrameParams {
  Sophus::SE3d T_c_w;
  AffineBrightness affine;
  double exposure = 1.0;
};

// Body pose and per-camera brightness of a rig keyframe.
struct KeyframeParams {
  Sophus::SE3d T_b_w;
  std::vector<AffineBrightness> affine;  // per camera
  std::vector<double> exposure;  // per camera

  FrameParams camera(const Rig& rig, int c) const { return {rig.T_c_b[c] * T_b_w, affine[c], exposure[c]}; }
};

struct WindowPixelResidual {
  double r = 0;
  double weight = 0;  // gradient weight * Huber IRLS weight
  // (xi, a, b): left increment on the host / target body pose, then the brightness of the host / target camera.
  Eigen::Matrix<double, 1, 8> dHost, dTarget;
  double dRho = 0;
};

struct WindowPatternResidual {
  std::array<WindowPixelResidual, kPatternSize> pixels;
  double energy = 0;
};

// Relative pose and brightness of a host/target camera pair at the current and at the linearisation parameters.
struct WindowPairState {
  Eigen::Matrix3d R, R0;  // T_t_h, T_t_h0
  Eigen::Vector3d t, t0;
  // Body increments: d(T_t_h) / d(host) = -adjHost0, d(T_t_h) / d(target) = adjTarget (left increments).
  Eigen::Matrix<double, 6, 6> adjHost0, adjTarget;
  double scale = 1, scale0 = 1;  // exposure ratio * exp(a_t - a_h)
  double hostB = 0, hostB0 = 0, targetB = 0;
  bool linearizedAtCurrent = false;
  bool identityTarget = true;  // adjTarget is the identity
  bool sameFrame = false;  // cameras of one keyframe: the residual does not depend on its pose
};

// Camera-level pair (single camera: extrinsics identity).
WindowPairState makeWindowPairState(const FrameParams& host, const FrameParams& target, const FrameParams& host0,
                                    const FrameParams& target0);
// Cameras of a rig; params are camera poses, increments are on the body poses behind them.
WindowPairState makeWindowPairState(const FrameParams& host, const FrameParams& target, const FrameParams& host0,
                                    const FrameParams& target0, const Sophus::SE3d& T_ch_b, const Sophus::SE3d& T_ct_b,
                                    bool sameFrame);

// Residual at the current parameters; frame Jacobians at the linearisation parameters host0/target0 (FEJ),
// image gradients at the current projection. DSO §2.3.
bool evaluateWindowResidual(const PatternPoint& point, double rho, const FrameParams& host,
                            const FrameParams& target, const FrameParams& host0, const FrameParams& target0,
                            const Camera& cam, const ImageLevel& targetImg, const PhotometricSettings& settings,
                            WindowPatternResidual& out);
bool evaluateWindowResidual(const PatternPoint& point, double rho, const WindowPairState& pair, const Camera& cam,
                            const ImageLevel& targetImg, const PhotometricSettings& settings,
                            WindowPatternResidual& out);
// Energy at the current parameters only; false if the pattern leaves the image.
bool windowResidualEnergy(const PatternPoint& point, double rho, const WindowPairState& pair, const Camera& cam,
                          const ImageLevel& targetImg, const PhotometricSettings& settings, double& energy);

struct WindowSettings {
  PhotometricSettings photometric;
  double outlierThreshold = 20.0;  // residual is an outlier if its energy exceeds that of |r| = threshold per pixel
  double affinePriorA = 1e3;
  double affinePriorB = 1.0;
  double firstFramePrior = 1e10;
  int maxIterations = 6;
  double minRhoHessian = 1e-3;  // points with less depth information are dropped instead of marginalised
  double stereoWeight = 1.0;  // relative weight of static residuals between cameras of the same keyframe
};

enum class ResidualState { Good, OutOfBounds, Outlier };

struct WindowResidual {
  int target;  // frame id
  int targetCam;
  ResidualState state = ResidualState::Good;
  double energy = 0;
};

struct WindowPoint {
  int id;
  int host;  // frame id
  int hostCam;
  PatternPoint pattern;
  Eigen::Vector3d bearing;  // in the host camera
  double rho;
  double hRho = 0;  // photometric depth information from the last optimisation
  // Every camera of every window keyframe except the host camera; same-keyframe ones are static residuals.
  std::vector<WindowResidual> residuals;

  int numGood() const;
};

struct WindowFrame {
  int id;
  std::vector<std::shared_ptr<const ImagePyramid>> images;  // per camera
  KeyframeParams params;
  KeyframeParams linearization;  // fixed once the frame is part of the marginalisation prior
  bool inPrior = false;

  Eigen::VectorXd delta() const;  // pose (6), then (a, b) per camera
};

struct WindowOptimizationResult {
  int iterations = 0;
  double initialEnergy = 0, finalEnergy = 0;
  int numGood = 0, numOutliers = 0, numOutOfBounds = 0;
};

// Sliding-window photometric bundle adjustment over keyframe body poses, per-camera affine brightness and point
// inverse distances, with Schur-complement marginalisation of old frames (DSO §2.3). Each keyframe has one image
// per rig camera; a point lives in one camera of one keyframe and has residuals in all other cameras.
class WindowOptimizer {
 public:
  WindowOptimizer(const Rig& rig, WindowSettings settings = {});
  WindowOptimizer(const Camera& cam, WindowSettings settings = {}) : WindowOptimizer(Rig::mono(cam), std::move(settings)) {}

  // One image per camera; brightness and exposure default to neutral.
  int addFrame(std::vector<std::shared_ptr<const ImagePyramid>> images, const Sophus::SE3d& T_b_w,
               std::vector<AffineBrightness> affine = {}, std::vector<double> exposure = {});
  int addFrame(std::shared_ptr<const ImagePyramid> image, const Sophus::SE3d& T_b_w, const AffineBrightness& affine = {},
               double exposure = 1.0) {
    return addFrame(std::vector{std::move(image)}, T_b_w, {affine}, {exposure});
  }
  // Returns the point id, or -1 if the pattern does not fit into the host image.
  int addPoint(int hostId, const Eigen::Vector2d& uv, double rho, int hostCam = 0);
  void removePoint(int pointId);
  void setFrameParams(int frameId, const KeyframeParams& params) { m_frames[frameIndex(frameId)].params = params; }

  WindowOptimizationResult optimize(int maxIterations = -1);
  // Calibration check for camera pairs with a common view: per temporal residual, the inverse depth minimising that
  // residual alone relative to the static (same-keyframe) optimum, averaged per image-radius third. A trend with
  // radius means a camera model error; it biases monocular depth and makes the scale drift.
  void logStaticDepthBias() const;
  void marginalizeFrame(int frameId);

  const std::vector<WindowFrame>& frames() const { return m_frames; }
  const std::vector<WindowPoint>& points() const { return m_points; }
  const WindowFrame& frame(int frameId) const { return m_frames[frameIndex(frameId)]; }
  const Rig& rig() const { return m_rig; }
  Sophus::SE3d cameraPose(int frameId, int cam) const { return m_rig.T_c_b[cam] * frame(frameId).params.T_b_w; }

 private:
  struct PointBlock {
    double Hrr = 0, gr = 0;
    Eigen::VectorXd Hfr;  // over all frame parameters
    std::vector<int> frames;  // frame indices with non-zero entries in Hfr
  };
  struct System {
    Eigen::MatrixXd H;
    Eigen::VectorXd g;
    std::vector<PointBlock> points;
  };
  // Pair states for the current frame parameters, by frame index and camera.
  struct Pairs {
    int numFrames = 0, numCams = 0;
    std::vector<WindowPairState> states;
    const WindowPairState& at(int host, int hostCam, int target, int targetCam) const {
      return states[((host * numCams + hostCam) * numFrames + target) * numCams + targetCam];
    }
  };

  int frameIndex(int frameId) const;
  int frameDim() const { return 6 + 2 * m_rig.size(); }
  // Rows of the pose and of camera c's brightness in the frame at index f.
  std::array<int, 8> parameterIndices(int f, int c) const;
  Pairs makePairs() const;
  // Accumulates the photometric system of the given points (all if empty) over Good residuals.
  System linearize(const Pairs& pairs, const std::vector<size_t>& pointIndices) const;
  void classifyResiduals();
  double energy(const Pairs& pairs) const;
  double pointEnergy(const Pairs& pairs, const WindowPoint& p, double rho) const;
  bool residualEnergy(const Pairs& pairs, const WindowPoint& p, const WindowResidual& r, double rho,
                      double& energy) const;
  double priorEnergy() const;
  void addPriors(System& sys) const;
  void schurPoint(const PointBlock& pb, double hr, Eigen::MatrixXd& H, Eigen::VectorXd& g) const;

  Rig m_rig;
  WindowSettings m_settings;
  std::vector<WindowFrame> m_frames;
  std::vector<WindowPoint> m_points;
  Eigen::MatrixXd m_priorH;  // over m_frames order, in delta coordinates
  Eigen::VectorXd m_priorB;
  int m_nextFrameId = 0;
  int m_nextPointId = 0;
};

}  // namespace sdv
