#pragma once

#include <array>
#include <optional>

#include <Eigen/Core>
#include <sophus/se3.hpp>

#include <sdv/camera.h>
#include <sdv/image_pyramid.h>
#include <sdv/photometric.h>

namespace sdv {

enum class TraceStatus { Uninitialized, Good, OutOfBounds, Outlier, Skipped, BadCondition, Ambiguous };

const char* toString(TraceStatus s);

struct TraceSettings {
  double rhoMaxInit = 5.0;
  double stepPixels = 1.0;
  int maxSamples = 100;
  // Coarse to fine: searches longer than coarseMinSamples steps sample every coarseStep-th step, then step by step
  // around the best match and the best one beyond secondBestExclusionPixels (1 = off).
  int coarseStep = 1;
  double coarseMinSamples = 16;
  double minSearchPixels = 1.5;
  double outlierEnergyPerPixel = 12.0 * 12.0;
  double minQuality = 2.0;
  double secondBestExclusionPixels = 2.0;
  int refineIterations = 4;
  double baseErrorPixels = 0.5;
  double maxErrorPixels = 8.0;
  PhotometricSettings photometric;
};

class ImmaturePoint {
 public:
  static std::optional<ImmaturePoint> create(const Camera& hostCam, const ImageLevel& hostImg,
                                             const Eigen::Vector2d& uv, const TraceSettings& settings);

  TraceStatus trace(const Camera& targetCam, const ImageLevel& targetImg, const HostTargetState& state,
                    const TraceSettings& settings);

  const PatternPoint& pattern() const { return m_pattern; }
  const Eigen::Vector3d& bearing() const { return m_bearing; }
  double rhoMin() const { return m_rhoMin; }
  double rhoMax() const { return m_rhoMax; }
  double rho() const { return m_rho; }
  double quality() const { return m_quality; }
  double lastErrorPixels() const { return m_errorPixels; }
  TraceStatus lastStatus() const { return m_lastStatus; }
  int numGood() const { return m_numGood; }
  int numOutliers() const { return m_numOutliers; }

 private:
  ImmaturePoint() = default;

  // offsets: the pattern warped into the target (warp * pattern offset), and scale: state.brightnessScale(), once per
  // trace.
  double patternEnergy(const Eigen::Vector2d& uv, const std::array<Eigen::Vector2d, kPatternSize>& offsets,
                       const ImageLevel& img, const HostTargetState& state, double scale, double huber) const;
  bool localWarp(const Camera& cam, double rho, const Sophus::SE3d& T_t_h, Eigen::Matrix2d& A) const;

  PatternPoint m_pattern;
  Eigen::Vector3d m_bearing;
  Eigen::Matrix2d m_structure = Eigen::Matrix2d::Zero();
  double m_rhoMin = 0, m_rhoMax = 0, m_rho = 0;
  double m_quality = 0, m_errorPixels = 0;
  TraceStatus m_lastStatus = TraceStatus::Uninitialized;
  int m_numGood = 0, m_numOutliers = 0;
};

}  // namespace sdv
