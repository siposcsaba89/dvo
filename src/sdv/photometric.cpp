#include <sdv/photometric.h>

#include <cmath>

namespace sdv {

double huberWeight(double r, double k) {
  const double a = std::abs(r);
  return a <= k ? 1.0 : k / a;
}

double huberEnergy(double r, double k) {
  const double a = std::abs(r);
  return a <= k ? 0.5 * r * r : k * (a - 0.5 * k);
}

std::optional<PatternPoint> makePatternPoint(const Camera& cam, const ImageLevel& img,
                                             const Eigen::Vector2d& uv, const PhotometricSettings& settings) {
  PatternPoint p;
  p.uv = uv;
  const double c2 = settings.gradientWeightC * settings.gradientWeightC;
  for (int k = 0; k < kPatternSize; ++k) {
    const Eigen::Vector2d q = uv + Eigen::Vector2d(kPattern[k][0], kPattern[k][1]);
    if (!cam.isInside(q.x(), q.y(), settings.border)) return std::nullopt;
    if (!cam.unproject(q, p.bearings[k])) return std::nullopt;
    const Eigen::Vector3f s = img.interpolate(static_cast<float>(q.x()), static_cast<float>(q.y()));
    p.intensities[k] = s[0];
    p.gradientWeights[k] = static_cast<float>(c2 / (c2 + s[1] * s[1] + s[2] * s[2]));
  }
  return p;
}

bool projectBearing(const Eigen::Vector3d& bearing, double rho, const Sophus::SE3d& T_t_h, const Camera& cam,
                    Eigen::Vector2d& uv, Eigen::Matrix<double, 2, 6>* dUvdPose, Eigen::Vector2d* dUvdRho) {
  const Eigen::Vector3d x = T_t_h.so3() * bearing + rho * T_t_h.translation();
  if (!dUvdPose && !dUvdRho) return cam.project(x, uv);

  Eigen::Matrix<double, 2, 3> J;
  if (!cam.project(x, uv, J)) return false;
  if (dUvdPose) {
    // d x / d xi for left increment: [rho * I, -[x]_x]
    dUvdPose->leftCols<3>() = rho * J;
    dUvdPose->rightCols<3>() = -J * Sophus::SO3d::hat(x);
  }
  if (dUvdRho) *dUvdRho = J * T_t_h.translation();
  return true;
}

bool evaluatePatternResidual(const PatternPoint& point, double rho, const HostTargetState& state,
                             const Camera& targetCam, const ImageLevel& targetImg,
                             const PhotometricSettings& settings, PatternResidual& out) {
  const double scale = state.brightnessScale();
  out.energy = 0;
  for (int k = 0; k < kPatternSize; ++k) {
    Eigen::Vector2d uv;
    Eigen::Matrix<double, 2, 6> dUvdPose;
    Eigen::Vector2d dUvdRho;
    if (!projectBearing(point.bearings[k], rho, state.T_t_h, targetCam, uv, &dUvdPose, &dUvdRho)) return false;
    if (!targetCam.isInside(uv.x(), uv.y(), settings.border)) return false;

    const Eigen::Vector3f s = targetImg.interpolate(static_cast<float>(uv.x()), static_cast<float>(uv.y()));
    const Eigen::RowVector2d grad(s[1], s[2]);
    const double hostCentered = point.intensities[k] - state.host.b;

    PixelResidual& px = out.pixels[k];
    px.r = (s[0] - state.target.b) - scale * hostCentered;
    px.dPose = grad * dUvdPose;
    px.dRho = grad * dUvdRho;
    px.dAffine << scale * hostCentered, scale, -scale * hostCentered, -1.0;
    px.weight = point.gradientWeights[k] * huberWeight(px.r, settings.huberThreshold);
    out.energy += point.gradientWeights[k] * huberEnergy(px.r, settings.huberThreshold);
  }
  return true;
}

}  // namespace sdv
