#pragma once

#include <array>
#include <cstdint>
#include <deque>
#include <memory>
#include <vector>

#include <opencv2/core.hpp>
#include <sophus/se3.hpp>

#include <sdv/immature_point.h>
#include <sdv/map_point.h>
#include <sdv/point_selector.h>
#include <sdv/rig.h>

namespace sdv {

struct SemiDenseSettings {
  int pointsPerImage = 20000;
  int traceFrames = 30;  // a host is traced into the images of this many following input frames
  double minDepth = 0.5;  // initial search range
  int minGood = 3;
  int dropFrames = 0;  // > 0: a candidate without a good trace this many frames after its host is not traced further
  double maxOutlierRatio = 0.2;  // outlier traces per good trace
  double maxInterval = 0.02;  // half width of the inverse-depth interval relative to the inverse depth
  // Multi-view consistency: points count per voxel of this size (world units); 0 = off.
  double voxelSize = 0.0;
  int minVoxelHosts = 1;  // a point is kept when its voxel holds points of this many different host images
  bool thin = false;  // keep only the most precise point per voxel
  // Multi-view verification when a host closes: the inverse depth is refined over all buffered views (host frame
  // and the traceFrames following ones, all cameras), then a point needs verifyMinViews views with parallax whose
  // pattern rmse is below verifyMaxError, and at least verifyMinFraction of its views with parallax so.
  bool verify = false;
  int verifyIterations = 3;
  double verifyMaxError = 10.0;  // intensity rmse per pattern pixel
  int verifyMinViews = 3;
  double verifyMinFraction = 0.5;
  double verifyMinParallax = 1.0;  // pixels a view moves for a 10 % inverse-depth change, to count
  TraceSettings trace;
};

// Semi-dense depth from known poses: every host image contributes many high-gradient pixels whose inverse-depth
// interval is narrowed by epipolar tracing into the other cameras of the same frame and into all cameras of the
// following frames, with the photometric error model of the odometry. No optimisation: poses and brightness are
// taken as given, so this runs after odometry has finished.
class SemiDenseMapper {
 public:
  SemiDenseMapper(const Rig& rig, SemiDenseSettings settings = {});

  // Frames in input order. images: prepared for the rig cameras (grey or BGR); T_w_b: final body pose; affine:
  // brightness per camera; host: whether this frame's images get new points (e.g. keyframes).
  void addFrame(int frameIndex, const std::vector<cv::Mat>& images, const Sophus::SE3d& T_w_b,
                const std::vector<AffineBrightness>& affine, bool host);
  // Appends the points to `out` (no copy of the cloud); the other form returns them.
  void finish(std::vector<MapPoint>& out);
  std::vector<MapPoint> finish() {
    std::vector<MapPoint> out;
    finish(out);
    return out;
  }

  struct Stats {
    long long traces = 0, good = 0, candidates = 0, accepted = 0, merged = 0;
    long long rejectMatches = 0, rejectInterval = 0, rejectVerify = 0;
    // Seconds: image pyramids, traces into later frames, new hosts (selection and the traces into the other cameras
    // of their frame), closing hosts.
    double pyramid = 0, trace = 0, create = 0, close = 0;
  };
  const Stats& stats() const { return m_stats; }

 private:
  struct Host {
    int frameIndex;
    int camera;
    Sophus::SE3d T_c_w;
    AffineBrightness affine;
    std::vector<ImmaturePoint> points;
    std::vector<float> intensity;
    std::vector<std::array<std::uint8_t, 3>> color;
    bool hasColor;
  };

  void createHosts(int frameIndex, const std::vector<cv::Mat>& images,
                   const std::vector<std::shared_ptr<const ImagePyramid>>& pyr, const std::vector<Sophus::SE3d>& T_c_w,
                   const std::vector<AffineBrightness>& affine);
  void traceFrame(int frameIndex, const std::vector<std::shared_ptr<const ImagePyramid>>& pyr, const std::vector<Sophus::SE3d>& T_c_w,
                  const std::vector<AffineBrightness>& affine);
  void tracePoint(ImmaturePoint& p, const Camera& cam, const ImageLevel& img, const HostTargetState& state,
                  bool requireVisible, long long& traces, long long& good) const;
  struct BufferedFrame {
    int frameIndex;
    std::vector<std::shared_ptr<const ImagePyramid>> pyr;
    std::vector<Sophus::SE3d> T_c_w;
    std::vector<AffineBrightness> affine;
  };

  void close(Host& h);
  // Refines rho over the buffered views; false if the point fails the verification.
  bool verify(const Host& h, const ImmaturePoint& p, double& rho) const;

  Rig m_rig;
  SemiDenseSettings m_settings;
  PointSelector m_selector;
  std::deque<Host> m_hosts;
  std::deque<BufferedFrame> m_frames;
  std::vector<MapPoint> m_points;
  Stats m_stats;
};

}  // namespace sdv
