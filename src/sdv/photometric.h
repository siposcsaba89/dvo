#pragma once

#include <array>
#include <cmath>
#include <optional>

#include <Eigen/Core>
#include <sophus/se3.hpp>

#include <sdv/camera.h>
#include <sdv/image_pyramid.h>

namespace sdv {

inline constexpr int kPatternSize = 8;
inline constexpr std::array<std::array<int, 2>, kPatternSize> kPattern = {
    {{-1, -1}, {1, -1}, {-1, 1}, {1, 1}, {0, -2}, {-2, 0}, {2, 0}, {0, 2}}};

struct AffineBrightness {
  double a = 0, b = 0;
};

struct PhotometricSettings {
  double huberThreshold = 9.0;
  double gradientWeightC = 50.0;
  double border = 2.0;
};

struct PatternPoint {
  Eigen::Vector2d uv;
  std::array<Eigen::Vector3d, kPatternSize> bearings;
  std::array<float, kPatternSize> intensities;
  std::array<float, kPatternSize> gradientWeights;
};

std::optional<PatternPoint> makePatternPoint(const Camera& cam, const ImageLevel& img,
                                             const Eigen::Vector2d& uv, const PhotometricSettings& settings);

struct HostTargetState {
  Sophus::SE3d T_t_h;
  AffineBrightness host, target;
  double exposureRatio = 1.0;  // t_target / t_host

  double brightnessScale() const { return exposureRatio * std::exp(target.a - host.a); }
};

// Target point is scaled by rho (X' = R*b + rho*t), which projects identically and allows rho = 0.
bool projectBearing(const Eigen::Vector3d& bearing, double rho, const Sophus::SE3d& T_t_h, const Camera& cam,
                    Eigen::Vector2d& uv, Eigen::Matrix<double, 2, 6>* dUvdPose = nullptr,
                    Eigen::Vector2d* dUvdRho = nullptr);
// Same with T_t_h as rotation matrix and translation, for many points per pose.
bool projectBearing(const Eigen::Vector3d& bearing, double rho, const Eigen::Matrix3d& R_t_h,
                    const Eigen::Vector3d& t_t_h, const Camera& cam, Eigen::Vector2d& uv,
                    Eigen::Matrix<double, 2, 6>* dUvdPose = nullptr, Eigen::Vector2d* dUvdRho = nullptr);

struct PixelResidual {
  double r = 0;
  double weight = 0;  // gradient weight * Huber IRLS weight
  Eigen::Matrix<double, 1, 6> dPose;
  double dRho = 0;
  Eigen::Matrix<double, 1, 4> dAffine;  // (a_h, b_h, a_t, b_t)
};

struct PatternResidual {
  std::array<PixelResidual, kPatternSize> pixels;
  double energy = 0;
};

bool evaluatePatternResidual(const PatternPoint& point, double rho, const HostTargetState& state,
                             const Camera& targetCam, const ImageLevel& targetImg,
                             const PhotometricSettings& settings, PatternResidual& out);

// T_t_h = T_t_w * T_h_w^-1 with left increments on world-to-camera poses.
inline Eigen::Matrix<double, 6, 6> dRelativeDTarget() { return Eigen::Matrix<double, 6, 6>::Identity(); }
inline Eigen::Matrix<double, 6, 6> dRelativeDHost(const Sophus::SE3d& T_t_h) { return -T_t_h.Adj(); }

inline double huberWeight(double r, double k) {
  const double a = std::abs(r);
  return a <= k ? 1.0 : k / a;
}
inline double huberEnergy(double r, double k) {
  const double a = std::abs(r);
  return a <= k ? 0.5 * r * r : k * (a - 0.5 * k);
}

}  // namespace sdv
