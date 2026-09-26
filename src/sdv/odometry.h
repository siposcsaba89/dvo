#pragma once

#include <map>
#include <memory>
#include <optional>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <sophus/se3.hpp>

#include <sdv/camera.h>
#include <sdv/immature_point.h>
#include <sdv/mono_initializer.h>
#include <sdv/point_selector.h>
#include <sdv/tracker.h>
#include <sdv/window_optimizer.h>

namespace sdv {

struct OdometrySettings {
  int levels = 5;
  MonoInitSettings init;
  double initMaxSigma = 0.02;  // relative depth sigma for initial points to become active
  TrackingSettings tracking;
  WindowSettings window;
  TraceSettings trace;
  int candidatesPerKeyframe = 1500;
  int maxKeyframes = 7;
  int targetActivePoints = 2000;
  int windowIterations = 6;
  double activationMaxErrorPixels = 2.0;
  int activationMinGood = 1;
  double initialActivationCell = 12.0;
  // New keyframe when flow / kfFlow + translation flow / kfTranslationFlow + |delta a| / kfBrightness > 1.
  double kfFlow = 120.0;
  double kfTranslationFlow = 50.0;
  double kfBrightness = 0.7;
  double kfRmseFactor = 2.0;
  double marginalizeVisibleFraction = 0.05;  // DSO §3.1
  double stereoMinDepth = 1.5;  // bounds the initial stereo search range
  int stereoMaxSamples = 400;
};

struct MapPoint {
  Eigen::Vector3d position;  // world
  float intensity;
  int frameIndex;  // input frame of the host keyframe
  Eigen::Vector2d uv;  // pixel in the host keyframe
  double distance;  // from the host camera
};

struct OdometryFrameInfo {
  bool initialized = false;
  bool keyframe = false;
  bool trackingOk = false;
  double rmse = 0;
  int activePoints = 0;
  int immaturePoints = 0;
};

// Direct sparse odometry: initialisation, frame tracking, candidate tracing, keyframe window optimisation and
// marginalisation. Monocular by default; with a stereo rig, keyframes add static stereo constraints (metric
// scale) and initialisation uses stereo depth.
class Odometry {
 public:
  Odometry(const Camera& cam, OdometrySettings settings = {});

  // T_r_l maps left-camera to right-camera coordinates. Must be called before the first frame.
  void enableStereo(const Camera& rightCam, const Sophus::SE3d& T_r_l);
  // `right` is required in stereo mode.
  OdometryFrameInfo addFrame(const cv::Mat& image, const cv::Mat& right = {});

  bool initialized() const { return m_initialized; }
  // T_w_c per input frame, final estimates; empty for frames before initialisation.
  std::vector<std::optional<Sophus::SE3d>> poses() const;
  // Marginalised points plus the active points still in the window.
  std::vector<MapPoint> mapPoints() const;
  std::vector<int> keyframeIndices() const;

 private:
  struct Keyframe {
    int frameIndex;
    std::shared_ptr<const ImagePyramid> image;
    std::vector<ImmaturePoint> immature;
  };
  struct FrameRecord {
    int keyframe = -1;  // optimizer frame id of the reference keyframe
    Sophus::SE3d T_f_kf;
  };

  void initializeFromMono(const MonoInitResult& res, std::shared_ptr<const ImagePyramid> current);
  int createKeyframe(std::shared_ptr<const ImagePyramid> image, const Sophus::SE3d& T_c_w,
                     const AffineBrightness& affine);
  void selectCandidates(int keyframeId);
  void traceStereo(int keyframeId, std::shared_ptr<const ImagePyramid> right);
  TraceSettings candidateSettings() const;
  void traceCandidates(const ImagePyramid& image, const Sophus::SE3d& T_c_w, const AffineBrightness& affine);
  void activateCandidates(int newKeyframeId);
  void removeOutlierPoints();
  void marginalizeKeyframes();
  void marginalize(int keyframeId);
  void buildReference();
  bool needKeyframe(const TrackingResult& res) const;
  double translationFlow(const Sophus::SE3d& T_f_ref) const;
  void storeKeyframePoses();

  Camera m_camera;
  OdometrySettings m_settings;
  MonoInitializer m_initializer;
  FrameTracker m_tracker;
  WindowOptimizer m_window;
  PointSelector m_selector;

  std::optional<Camera> m_rightCam;
  Sophus::SE3d m_T_r_l;
  cv::Mat m_currentRight;

  bool m_initialized = false;
  int m_frameCount = 0;
  std::shared_ptr<const ImagePyramid> m_initHost;
  int m_initHostIndex = -1;

  std::map<int, Keyframe> m_keyframes;  // by optimizer frame id, window only
  std::map<int, Sophus::SE3d> m_keyframePoses;  // T_c_w, latest estimate, also marginalised ones
  std::map<int, int> m_keyframeFrameIndex;
  std::vector<FrameRecord> m_frames;
  std::vector<MapPoint> m_marginalizedPoints;

  std::optional<ReferenceFrame> m_reference;
  int m_referenceId = -1;
  AffineBrightness m_lastAffine;
  Sophus::SE3d m_T_prev_ref, m_T_prev_prevprev;
  double m_referenceRmse = -1;
  double m_activationCell;
};

}  // namespace sdv
