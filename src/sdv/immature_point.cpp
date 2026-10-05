#include <sdv/immature_point.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace sdv {

const char* toString(TraceStatus s) {
  switch (s) {
    case TraceStatus::Uninitialized: return "uninitialized";
    case TraceStatus::Good: return "good";
    case TraceStatus::OutOfBounds: return "oob";
    case TraceStatus::Outlier: return "outlier";
    case TraceStatus::Skipped: return "skipped";
    case TraceStatus::BadCondition: return "badcondition";
    case TraceStatus::Ambiguous: return "ambiguous";
  }
  return "?";
}

std::optional<ImmaturePoint> ImmaturePoint::create(const Camera& hostCam, const ImageLevel& hostImg,
                                                   const Eigen::Vector2d& uv, const TraceSettings& settings) {
  auto pattern = makePatternPoint(hostCam, hostImg, uv, settings.photometric);
  if (!pattern) return std::nullopt;
  ImmaturePoint p;
  if (!hostCam.unproject(uv, p.m_bearing)) return std::nullopt;
  p.m_pattern = *pattern;
  for (int k = 0; k < kPatternSize; ++k) {
    const Eigen::Vector3f s = hostImg.interpolate(float(uv.x() + kPattern[k][0]), float(uv.y() + kPattern[k][1]));
    const Eigen::Vector2d g(s[1], s[2]);
    p.m_structure += g * g.transpose();
  }
  p.m_rhoMin = 0.0;
  p.m_rhoMax = settings.rhoMaxInit;
  p.m_rho = 0.5 * settings.rhoMaxInit;
  return p;
}

bool ImmaturePoint::localWarp(const Camera& cam, double rho, const Sophus::SE3d& T_t_h, Eigen::Matrix2d& A) const {
  // Least-squares affine map of host pattern offsets to target offsets.
  Eigen::Vector2d center;
  if (!projectBearing(m_bearing, rho, T_t_h, cam, center)) return false;
  Eigen::Matrix2d DtD = Eigen::Matrix2d::Zero(), PtD = Eigen::Matrix2d::Zero();
  for (int k = 0; k < kPatternSize; ++k) {
    Eigen::Vector2d uv;
    if (!projectBearing(m_pattern.bearings[k], rho, T_t_h, cam, uv)) return false;
    const Eigen::Vector2d d(kPattern[k][0], kPattern[k][1]);
    DtD += d * d.transpose();
    PtD += (uv - center) * d.transpose();
  }
  A = PtD * DtD.inverse();
  return true;
}

double ImmaturePoint::patternEnergy(const Eigen::Vector2d& uv, const Eigen::Matrix2d& warp, const ImageLevel& img,
                                    const HostTargetState& state, double huber) const {
  const double scale = state.brightnessScale();
  double e = 0;
  for (int k = 0; k < kPatternSize; ++k) {
    const Eigen::Vector2d q = uv + warp * Eigen::Vector2d(kPattern[k][0], kPattern[k][1]);
    if (q.x() < 1 || q.y() < 1 || q.x() > img.width - 3 || q.y() > img.height - 3)
      return std::numeric_limits<double>::infinity();
    const double r = (img.interpolateIntensity(float(q.x()), float(q.y())) - state.target.b) -
                     scale * (m_pattern.intensities[k] - state.host.b);
    e += huberEnergy(r, huber);
  }
  return e;
}

TraceStatus ImmaturePoint::trace(const Camera& cam, const ImageLevel& img, const HostTargetState& state,
                                 const TraceSettings& settings) {
  const Sophus::SE3d& T = state.T_t_h;
  const double huber = settings.photometric.huberThreshold;
  const double border = settings.photometric.border + 2.0;
  auto finish = [&](TraceStatus s) {
    m_lastStatus = s;
    if (s == TraceStatus::Outlier) ++m_numOutliers;
    return s;
  };

  double rMin = m_rhoMin, rMax = m_rhoMax;
  Eigen::Vector2d uvMin, uvMax;
  if (!projectBearing(m_bearing, rMin, T, cam, uvMin)) return finish(TraceStatus::OutOfBounds);
  bool okMax = projectBearing(m_bearing, rMax, T, cam, uvMax);
  for (int i = 0; i < 20 && !okMax; ++i) {
    rMax = 0.5 * (rMin + rMax);
    okMax = projectBearing(m_bearing, rMax, T, cam, uvMax);
  }
  if (!okMax) return finish(TraceStatus::OutOfBounds);

  const double length = (uvMax - uvMin).norm();
  if (length < settings.minSearchPixels) {
    const bool visible = cam.isInside(uvMin.x(), uvMin.y(), border);
    return finish(visible ? TraceStatus::Skipped : TraceStatus::OutOfBounds);
  }

  Eigen::Matrix2d A;
  const double rhoGuess = m_numGood > 0 ? std::clamp(m_rho, rMin, rMax) : 0.5 * (rMin + rMax);
  if (!localWarp(cam, rhoGuess, T, A)) return finish(TraceStatus::OutOfBounds);

  struct Sample {
    double rho;
    Eigen::Vector2d uv;
    double energy;
    double rhoStep;
  };
  thread_local std::vector<Sample> samples;  // reused: traces run millions of times per frame in the densify
  samples.clear();
  const double step = std::max(settings.stepPixels, length / settings.maxSamples);
  for (double rho = rMin; rho <= rMax && samples.size() <= size_t(settings.maxSamples) * 2;) {
    Eigen::Vector2d uv, dRho;
    if (!projectBearing(m_bearing, rho, T, cam, uv, nullptr, &dRho)) break;
    const double speed = dRho.norm();
    if (speed < 1e-9) break;
    const double rhoStep = step / speed;
    if (cam.isInside(uv.x(), uv.y(), border)) samples.push_back({rho, uv, patternEnergy(uv, A, img, state, huber), rhoStep});
    rho += rhoStep;
  }
  if (samples.empty()) return finish(TraceStatus::OutOfBounds);

  const auto bestIt = std::min_element(samples.begin(), samples.end(),
                                       [](const Sample& a, const Sample& b) { return a.energy < b.energy; });
  Sample best = *bestIt;
  double secondBest = std::numeric_limits<double>::infinity();
  for (const auto& s : samples)
    if ((s.uv - best.uv).norm() > settings.secondBestExclusionPixels) secondBest = std::min(secondBest, s.energy);

  const double scale = state.brightnessScale();
  for (int it = 0; it < settings.refineIterations; ++it) {
    Eigen::Vector2d uv, dRho;
    if (!projectBearing(m_bearing, best.rho, T, cam, uv, nullptr, &dRho)) break;
    double H = 0, b = 0;
    for (int k = 0; k < kPatternSize; ++k) {
      const Eigen::Vector2d q = uv + A * Eigen::Vector2d(kPattern[k][0], kPattern[k][1]);
      if (!cam.isInside(q.x(), q.y(), 1.0)) {
        H = 0;
        break;
      }
      const Eigen::Vector3f s = img.interpolate(float(q.x()), float(q.y()));
      const double r = (s[0] - state.target.b) - scale * (m_pattern.intensities[k] - state.host.b);
      const double J = Eigen::Vector2d(s[1], s[2]).dot(dRho);
      const double w = huberWeight(r, huber);
      H += w * J * J;
      b += w * J * r;
    }
    if (H <= 1e-12) break;
    const double delta = std::clamp(-b / H, -best.rhoStep, best.rhoStep);
    const double rho = std::clamp(best.rho + delta, rMin, rMax);
    Eigen::Vector2d uvNew;
    if (!projectBearing(m_bearing, rho, T, cam, uvNew) || !cam.isInside(uvNew.x(), uvNew.y(), border)) break;
    const double e = patternEnergy(uvNew, A, img, state, huber);
    if (e >= best.energy) break;
    best = {rho, uvNew, e, best.rhoStep};
  }

  m_quality = best.energy > 1e-9 ? secondBest / best.energy : 1e9;
  if (best.energy > settings.outlierEnergyPerPixel * kPatternSize) return finish(TraceStatus::Outlier);

  Eigen::Vector2d uv, dRho;
  if (!projectBearing(m_bearing, best.rho, T, cam, uv, nullptr, &dRho) || dRho.norm() < 1e-9)
    return finish(TraceStatus::BadCondition);

  // Matching precision along the search direction, from the host structure tensor warped into the target.
  const Eigen::Matrix2d Ainv = A.inverse();
  const Eigen::Matrix2d M = Ainv.transpose() * m_structure * Ainv;
  const Eigen::Vector2d dir = dRho.normalized();
  const double along = dir.dot(M * dir);
  m_errorPixels = settings.baseErrorPixels * std::sqrt(M.trace() / std::max(along, 1e-3 * M.trace() + 1e-12));
  if (m_errorPixels > settings.maxErrorPixels) return finish(TraceStatus::BadCondition);
  if (m_quality < settings.minQuality) return finish(TraceStatus::Ambiguous);

  const double dr = m_errorPixels / dRho.norm();
  const double newMin = std::max(best.rho - dr, 0.0), newMax = best.rho + dr;
  m_rhoMin = std::max(newMin, m_rhoMin - dr);
  m_rhoMax = std::min(newMax, m_rhoMax + dr);
  if (m_rhoMin > m_rhoMax) {
    m_rhoMin = newMin;
    m_rhoMax = newMax;
  }
  m_rho = best.rho;
  ++m_numGood;
  return finish(TraceStatus::Good);
}

}  // namespace sdv
