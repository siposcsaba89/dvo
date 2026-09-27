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
#include <sdv/map_point.h>
#include <sdv/mono_initializer.h>
#include <sdv/point_selector.h>
#include <sdv/profiler.h>
#include <sdv/rig.h>
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
  int candidatesPerKeyframe = 1500;  // per camera
  int maxKeyframes = 7;
  int targetActivePoints = 2000;  // per camera
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
  double stereoMinDepth = 1.5;  // bounds the initial search range between cameras of one keyframe
  int stereoMaxSamples = 400;
  bool checkCalibration = false;  // multi-camera: log the temporal vs static depth bias at every keyframe
  // Converged candidates that were never activated also become map points when their keyframe leaves the window.
  bool mapCandidates = false;
  int candidateMinGood = 2;
  double candidateMaxInterval = 0.05;  // half width of the inverse-depth interval relative to the inverse depth
};

struct OdometryFrameInfo {
  bool initialized = false;
  bool keyframe = false;
  bool trackingOk = false;
  double rmse = 0;
  int activePoints = 0;
  int immaturePoints = 0;
};

// Direct sparse odometry for a rig of one or more cameras: initialisation, frame tracking, candidate tracing,
// keyframe window optimisation and marginalisation. A single camera initialises monocularly; with several cameras,
// candidates are matched between cameras of the same keyframe, which gives metric depth from the extrinsics.
class Odometry {
 public:
  Odometry(const Rig& rig, OdometrySettings settings = {});
  Odometry(const Camera& cam, OdometrySettings settings = {}) : Odometry(Rig::mono(cam), std::move(settings)) {}

  // One image per rig camera, taken at the same time.
  OdometryFrameInfo addFrame(const std::vector<cv::Mat>& images);
  OdometryFrameInfo addFrame(const cv::Mat& image) { return addFrame(std::vector{image}); }

  bool initialized() const { return m_initialized; }
  // T_w_b per input frame, final estimates; empty for frames before initialisation.
  std::vector<std::optional<Sophus::SE3d>> poses() const;
  // Affine brightness per camera and input frame: keyframes from the window, other frames from tracking.
  std::vector<std::vector<AffineBrightness>> brightness() const;
  // Marginalised points plus the active points still in the window (and converged candidates, see mapCandidates).
  std::vector<MapPoint> mapPoints() const;
  std::vector<int> keyframeIndices() const;
  const Rig& rig() const { return m_rig; }
  const StageProfile& profile() const { return m_profile; }

 private:
  using Pyramids = std::vector<std::shared_ptr<const ImagePyramid>>;
  struct Keyframe {
    int frameIndex;
    Pyramids images;
    std::vector<std::vector<ImmaturePoint>> immature;  // per host camera
  };
  struct FrameRecord {
    int keyframe = -1;  // optimizer frame id of the reference keyframe
    Sophus::SE3d T_f_kf;  // body
    std::vector<AffineBrightness> affine;  // tracked frames
  };

  bool multiCamera() const { return m_rig.size() > 1; }
  void initializeFromMono(const MonoInitResult& res, const Pyramids& current);
  int createKeyframe(const Pyramids& images, const Sophus::SE3d& T_b_w, const std::vector<AffineBrightness>& affine);
  void selectCandidates(int keyframeId);
  void traceStatic(int keyframeId);
  void estimateStaticBrightness(int keyframeId);
  TraceSettings candidateSettings() const;
  void traceCandidates(const Pyramids& images, const Sophus::SE3d& T_b_w, const std::vector<AffineBrightness>& affine);
  void activateCandidates(int newKeyframeId);
  void removeOutlierPoints();
  void marginalizeKeyframes();
  void marginalize(int keyframeId);
  void buildReference();
  bool needKeyframe(const TrackingResult& res) const;
  double translationFlow(const Sophus::SE3d& T_f_ref) const;
  void storeKeyframePoses();
  MapPoint mapPoint(const WindowPoint& p) const;
  void appendCandidatePoints(int keyframeId, std::vector<MapPoint>& out) const;

  Rig m_rig;
  OdometrySettings m_settings;
  MonoInitializer m_initializer;
  FrameTracker m_tracker;
  WindowOptimizer m_window;
  PointSelector m_selector;

  bool m_initialized = false;
  int m_frameCount = 0;
  std::shared_ptr<const ImagePyramid> m_initHost;
  int m_initHostIndex = -1;

  std::map<int, Keyframe> m_keyframes;  // by optimizer frame id, window only
  std::map<int, Sophus::SE3d> m_keyframePoses;  // T_b_w, latest estimate, also marginalised ones
  std::map<int, std::vector<AffineBrightness>> m_keyframeAffine;
  std::map<int, int> m_keyframeFrameIndex;
  std::vector<FrameRecord> m_frames;
  std::vector<MapPoint> m_marginalizedPoints;

  std::vector<ReferenceFrame> m_reference;  // per camera
  int m_referenceId = -1;
  std::vector<AffineBrightness> m_lastAffine;
  Sophus::SE3d m_T_prev_ref, m_T_prev_prevprev;
  double m_referenceRmse = -1;
  double m_activationCell;
  StageProfile m_profile;
};

}  // namespace sdv
