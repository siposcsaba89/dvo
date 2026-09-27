#include <sdv/tracker.h>

#include <algorithm>
#include <cmath>
#include <execution>
#include <limits>
#include <numeric>

#include <Eigen/Cholesky>

#include <sdv/parallel.h>

namespace sdv {

ReferenceFrame::ReferenceFrame(const Camera& cam, const ImagePyramid& pyramid,
                               const std::vector<Eigen::Vector2i>& pixels, const std::vector<double>& rho,
                               const AffineBrightness& affine, double gradientWeightC)
    : m_affine(affine) {
  const int numLevels = pyramid.numLevels();
  m_points.resize(numLevels);
  m_cameras.resize(numLevels);
  const double c2 = gradientWeightC * gradientWeightC;

  std::vector<Eigen::Vector2i> levelPixels = pixels;
  std::vector<double> levelRho = rho;
  for (int l = 0; l < numLevels; ++l) {
    m_cameras[l] = cam.atLevel(l);
    const ImageLevel& img = pyramid.level(l);

    if (l > 0) {
      const int w = img.width, h = img.height;
      std::vector<double> sum(static_cast<size_t>(w) * h, 0.0);
      std::vector<int> count(sum.size(), 0);
      for (size_t i = 0; i < levelPixels.size(); ++i) {
        const int u = levelPixels[i].x() / 2, v = levelPixels[i].y() / 2;
        if (u >= w || v >= h) continue;
        sum[v * w + u] += levelRho[i];
        ++count[v * w + u];
      }
      levelPixels.clear();
      levelRho.clear();
      for (int v = 0; v < h; ++v)
        for (int u = 0; u < w; ++u)
          if (count[v * w + u] > 0) {
            levelPixels.emplace_back(u, v);
            levelRho.push_back(sum[v * w + u] / count[v * w + u]);
          }
    }

    auto& pts = m_points[l];
    pts.reserve(levelPixels.size());
    for (size_t i = 0; i < levelPixels.size(); ++i) {
      const Eigen::Vector2d uv = levelPixels[i].cast<double>();
      if (!m_cameras[l].isInside(uv.x(), uv.y(), 1.0)) continue;
      Point p;
      if (!m_cameras[l].unproject(uv, p.bearing)) continue;
      const Eigen::Vector3f& s = img.at(levelPixels[i].x(), levelPixels[i].y());
      p.uv = uv;
      p.rho = levelRho[i];
      p.intensity = s[0];
      p.gradientWeight = static_cast<float>(c2 / (c2 + s[1] * s[1] + s[2] * s[2]));
      pts.push_back(p);
    }
  }
}

FrameTracker::System FrameTracker::linearize(const ReferenceFrame& ref, const ImageLevel& img, int level,
                                             const State& state, double cutoff, double exposureRatio,
                                             bool withJacobians) const {
  const Camera& cam = ref.camera(level);
  const AffineBrightness& host = ref.affine();
  const double scale = exposureRatio * std::exp(state.affine.a - host.a);
  const double k = m_settings.huberThreshold;
  const double cutoffEnergy = huberEnergy(cutoff, k);

  const Eigen::Matrix3d R = state.T_t_h.rotationMatrix();
  const Eigen::Vector3d t = state.T_t_h.translation();
  const auto& points = ref.points(level);
  std::vector<System> chunks(kParallelChunks);
  parallelChunks(points.size(), [&](size_t c, size_t begin, size_t end) {
    System& part = chunks[c];
    part.H.setZero();
    part.g.setZero();
    Eigen::Matrix<double, 1, 8> J;
    Eigen::Vector2d uv;
    Eigen::Matrix<double, 2, 6> dUv;
    for (size_t i = begin; i < end; ++i) {
      const ReferenceFrame::Point& p = points[i];
      ++part.numPoints;
      if (!projectBearing(p.bearing, p.rho, R, t, cam, uv, withJacobians ? &dUv : nullptr) ||
          !cam.isInside(uv.x(), uv.y(), m_settings.border))
        continue;
      ++part.numVisible;
      const Eigen::Vector3f s = img.interpolate(static_cast<float>(uv.x()), static_cast<float>(uv.y()));
      const double hostCentered = p.intensity - host.b;
      const double r = (s[0] - state.affine.b) - scale * hostCentered;
      if (std::abs(r) > cutoff) {
        part.energy += p.gradientWeight * cutoffEnergy;
        continue;
      }
      ++part.numInliers;
      part.energy += p.gradientWeight * huberEnergy(r, k);
      if (!withJacobians) continue;
      J.head<6>() = Eigen::RowVector2d(s[1], s[2]) * dUv;
      J[6] = -scale * hostCentered;
      J[7] = -1.0;
      const double w = p.gradientWeight * huberWeight(r, k);
      part.H.noalias() += w * J.transpose() * J;
      part.g.noalias() += w * r * J.transpose();
    }
  });

  System sys;
  sys.H.setZero();
  sys.g.setZero();
  for (const System& part : chunks) {
    sys.H += part.H;
    sys.g += part.g;
    sys.energy += part.energy;
    sys.numInliers += part.numInliers;
    sys.numVisible += part.numVisible;
    sys.numPoints += part.numPoints;
  }

  const double priors[2] = {m_settings.affinePriorA, m_settings.affinePriorB};
  const double deltas[2] = {state.affine.a - host.a, state.affine.b - host.b};
  for (int i = 0; i < 2; ++i) {
    if (priors[i] <= 0) continue;
    sys.H(6 + i, 6 + i) += priors[i];
    sys.g[6 + i] += priors[i] * deltas[i];
    sys.energy += 0.5 * priors[i] * deltas[i] * deltas[i];
  }
  return sys;
}

FrameTracker::System FrameTracker::optimizeLevel(const ReferenceFrame& ref, const ImageLevel& img, int level,
                                                 State& state, double cutoff, double exposureRatio, int maxIt) const {
  System sys = linearize(ref, img, level, state, cutoff, exposureRatio, true);
  double lambda = 1e-3;
  for (int it = 0; it < maxIt && sys.numInliers > 8; ++it) {
    Eigen::Matrix<double, 8, 8> A = sys.H;
    A.diagonal() *= 1.0 + lambda;
    A.diagonal().array() += 1e-9;
    const Eigen::Matrix<double, 8, 1> delta = -A.ldlt().solve(sys.g);

    State candidate;
    candidate.T_t_h = Sophus::SE3d::exp(delta.head<6>()) * state.T_t_h;
    candidate.affine = {state.affine.a + delta[6], state.affine.b + delta[7]};
    const System next = linearize(ref, img, level, candidate, cutoff, exposureRatio, false);
    if (next.numInliers > 8 && next.normalizedEnergy() < sys.normalizedEnergy()) {
      state = candidate;
      sys = linearize(ref, img, level, state, cutoff, exposureRatio, true);
      lambda = std::max(lambda * 0.5, 1e-7);
      if (delta.head<6>().norm() < m_settings.convergenceEps) break;
    } else {
      lambda *= 4.0;
      if (lambda > 1e4) break;
    }
  }
  return sys;
}

double FrameTracker::meanFlow(const ReferenceFrame& ref, const State& state) const {
  double sum = 0;
  int n = 0;
  Eigen::Vector2d uv;
  for (const auto& p : ref.points(0)) {
    if (!projectBearing(p.bearing, p.rho, state.T_t_h, ref.camera(0), uv)) continue;
    sum += (uv - p.uv).norm();
    ++n;
  }
  return n > 0 ? sum / n : 0.0;
}

TrackingResult FrameTracker::track(const ReferenceFrame& ref, const ImagePyramid& target,
                                   const std::vector<Sophus::SE3d>& hypotheses,
                                   const AffineBrightness& initialAffine, double exposureRatio) const {
  TrackingResult result;
  const int top = std::min(ref.numLevels(), target.numLevels()) - 1;
  if (hypotheses.empty() || top < 0) return result;

  auto iterations = [&](int level) {
    return m_settings.maxIterations[std::min<size_t>(level, m_settings.maxIterations.size() - 1)];
  };

  std::vector<State> states(hypotheses.size());
  std::vector<System> systems(hypotheses.size());
  std::vector<size_t> order(hypotheses.size());
  std::iota(order.begin(), order.end(), size_t{0});
  std::for_each(std::execution::par, order.begin(), order.end(), [&](size_t i) {
    states[i] = {hypotheses[i], initialAffine};
    systems[i] = optimizeLevel(ref, target.level(top), top, states[i], m_settings.outlierCutoff, exposureRatio,
                               m_settings.hypothesisIterations);
  });
  State best;
  double bestRmse = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < hypotheses.size(); ++i) {
    const System& sys = systems[i];
    const double rmse = std::sqrt(sys.normalizedEnergy());
    if (sys.numVisible >= m_settings.minVisibleRatio * sys.numPoints && rmse < bestRmse) {
      bestRmse = rmse;
      best = states[i];
      result.hypothesis = static_cast<int>(i);
    }
  }

  System sys;
  for (int l = top; l >= 0; --l) {
    double cutoff = m_settings.outlierCutoff;
    for (int attempt = 0;; ++attempt) {
      State s = best;
      sys = optimizeLevel(ref, target.level(l), l, s, cutoff, exposureRatio, iterations(l));
      if (sys.inlierRatio() >= m_settings.minInlierRatio || attempt >= m_settings.maxCutoffIncreases) {
        best = s;
        break;
      }
      cutoff *= 2.0;
    }
  }

  result.T_t_h = best.T_t_h;
  result.affine = best.affine;
  result.rmse = std::sqrt(sys.normalizedEnergy());
  result.inlierRatio = sys.inlierRatio();
  result.visibleRatio = sys.numPoints > 0 ? double(sys.numVisible) / sys.numPoints : 0.0;
  result.meanFlow = meanFlow(ref, best);
  result.ok = result.hypothesis >= 0 && result.inlierRatio >= m_settings.minInlierRatio &&
              result.visibleRatio >= m_settings.minVisibleRatio;
  return result;
}

std::vector<Sophus::SE3d> makeMotionHypotheses(const Sophus::SE3d& T_prev_ref, const Sophus::SE3d& T_prev_prevprev) {
  const Sophus::Vector6d v = T_prev_prevprev.log();
  const Sophus::SE3d constVel = T_prev_prevprev * T_prev_ref;
  std::vector<Sophus::SE3d> h = {
      constVel,
      Sophus::SE3d::exp(2.0 * v) * T_prev_ref,
      Sophus::SE3d::exp(0.5 * v) * T_prev_ref,
      T_prev_ref,
  };
  for (double angle : {0.01, 0.03}) {
    for (int axis = 0; axis < 3; ++axis) {
      for (double sign : {1.0, -1.0}) {
        Sophus::Vector6d d = Sophus::Vector6d::Zero();
        d[3 + axis] = sign * angle;
        h.push_back(Sophus::SE3d::exp(d) * constVel);
      }
    }
  }
  return h;
}

}  // namespace sdv
