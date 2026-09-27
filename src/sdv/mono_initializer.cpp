#include <sdv/mono_initializer.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include <Eigen/Cholesky>

#include <sdv/parallel.h>
#include <sdv/point_selector.h>

namespace sdv {

namespace {

constexpr double kMinRho = 1e-4;

}  // namespace

MonoInitializer::MonoInitializer(const Camera& cam, int numLevels, MonoInitSettings settings)
    : m_camera(cam), m_numLevels(numLevels), m_settings(std::move(settings)) {}

void MonoInitializer::reset(const ImagePyramid& first, const cv::Mat& rhoPrior) {
  const int numLevels = std::min(m_numLevels, first.numLevels());
  m_levels.assign(numLevels, {});
  for (int l = 0; l < numLevels; ++l) {
    Level& level = m_levels[l];
    level.cam = m_camera.atLevel(l);
    PointSelectorSettings sel;
    sel.targetPoints =
        std::max(20, static_cast<int>(m_settings.pointsLevel0 * std::pow(m_settings.levelPointFactor, l)));
    for (const auto& c : PointSelector(sel).select(first.level(l), level.cam.maskImage())) {
      const Eigen::Vector2d uv = c.uv.cast<double>();
      auto pattern = makePatternPoint(level.cam, first.level(l), uv, m_settings.photometric);
      Point p;
      if (!pattern || !level.cam.unproject(uv, p.bearing)) continue;
      p.pattern = *pattern;
      p.rho = rhoPrior.empty() ? 1.0 : 0.0;
      if (!rhoPrior.empty()) {
        const int s = 1 << l;
        const int u0 = std::clamp(static_cast<int>(std::lround((uv.x() + 0.5) * s - 0.5)), 0, rhoPrior.cols - 1);
        const int v0 = std::clamp(static_cast<int>(std::lround((uv.y() + 0.5) * s - 0.5)), 0, rhoPrior.rows - 1);
        if (const float r = rhoPrior.at<float>(v0, u0); r > 0) p.rho = r;
      }
      for (float w : p.pattern.gradientWeights)
        p.energy += w * huberEnergy(m_settings.outlierCutoff, m_settings.photometric.huberThreshold);
      level.points.push_back(std::move(p));
    }

    auto& pts = level.points;
    const int k = std::min<int>(m_settings.numNeighbours, static_cast<int>(pts.size()) - 1);
    std::vector<std::pair<double, int>> dist;
    for (size_t i = 0; i < pts.size() && k > 0; ++i) {
      dist.clear();
      for (size_t j = 0; j < pts.size(); ++j)
        if (j != i) dist.emplace_back((pts[j].pattern.uv - pts[i].pattern.uv).squaredNorm(), static_cast<int>(j));
      std::partial_sort(dist.begin(), dist.begin() + k, dist.end());
      for (int n = 0; n < k; ++n) pts[i].neighbours.push_back(dist[n].second);
    }
  }

  for (int l = 0; l + 1 < numLevels; ++l) {
    const auto& coarse = m_levels[l + 1].points;
    for (auto& p : m_levels[l].points) {
      const Eigen::Vector2d uvCoarse = (p.pattern.uv.array() - 0.5) / 2.0;
      double best = std::numeric_limits<double>::infinity();
      for (size_t j = 0; j < coarse.size(); ++j) {
        const double d = (coarse[j].pattern.uv - uvCoarse).squaredNorm();
        if (d < best) {
          best = d;
          p.parent = static_cast<int>(j);
        }
      }
    }
  }

  double sum = 0;
  int count = 0;
  for (const auto& p : m_levels[0].points)
    if (p.rho > 0) {
      sum += p.rho;
      ++count;
    }
  m_gaugeRho = count > 0 ? sum / count : 1.0;
  for (auto& level : m_levels)
    for (auto& p : level.points)
      if (p.rho <= 0) p.rho = m_gaugeRho;
  m_state = m_prevState = {};
  m_numFrames = 1;
}

MonoInitializer::System MonoInitializer::linearize(int level, const ImageLevel& img, const State& state,
                                                   const std::vector<double>& rho,
                                                   const std::vector<double>& fallbackEnergy,
                                                   bool withJacobians) const {
  const auto& pts = m_levels[level].points;
  const Camera& cam = m_levels[level].cam;
  const size_t n = pts.size();
  const double k = m_settings.photometric.huberThreshold;
  const double cutoff = m_settings.outlierCutoff;
  const double cutoffEnergy = huberEnergy(cutoff, k);

  System sys;
  sys.Hxx.setZero();
  sys.gx.setZero();
  sys.valid.assign(n, 0);
  sys.pointEnergy.assign(n, 0.0);
  sys.pointPrior.assign(n, 0.0);
  if (withJacobians) {
    sys.Hxr.assign(n, Eigen::Matrix<double, 8, 1>::Zero());
    sys.Hrr.assign(n, 0.0);
    sys.gr.assign(n, 0.0);
    sys.hPhoto.assign(n, 0.0);
  }

  HostTargetState hts;
  hts.T_t_h = state.T_t_h;
  hts.target = state.affine;
  struct Partial {
    Eigen::Matrix<double, 8, 8> Hxx = Eigen::Matrix<double, 8, 8>::Zero();
    Eigen::Matrix<double, 8, 1> gx = Eigen::Matrix<double, 8, 1>::Zero();
    int numResiduals = 0, numVisible = 0, numInliers = 0;
    double visibleEnergy = 0, photometricEnergy = 0, priorEnergy = 0;
  };
  std::vector<Partial> partials(kParallelChunks);
  parallelChunks(n, [&](size_t c, size_t begin, size_t end) {
    Partial& part = partials[c];
    PatternResidual res;
    Eigen::Matrix<double, 8, 1> Jx;
    for (size_t i = begin; i < end; ++i) {
      const Point& p = pts[i];
      if (evaluatePatternResidual(p.pattern, rho[i], hts, cam, img, m_settings.photometric, res)) {
        sys.valid[i] = 1;
        part.numResiduals += kPatternSize;
        part.numVisible += 1;
        for (int q = 0; q < kPatternSize; ++q) {
          const PixelResidual& px = res.pixels[q];
          const double wg = p.pattern.gradientWeights[q];
          if (std::abs(px.r) > cutoff) {
            sys.pointEnergy[i] += wg * cutoffEnergy;
            continue;
          }
          ++part.numInliers;
          sys.pointEnergy[i] += wg * huberEnergy(px.r, k);
          if (!withJacobians) continue;
          Jx << px.dPose.transpose(), px.dAffine[2], px.dAffine[3];
          const double w = px.weight;
          part.Hxx.noalias() += w * Jx * Jx.transpose();
          part.gx.noalias() += w * px.r * Jx;
          sys.Hxr[i].noalias() += w * px.dRho * Jx;
          sys.Hrr[i] += w * px.dRho * px.dRho;
          sys.gr[i] += w * px.r * px.dRho;
        }
        if (withJacobians) sys.hPhoto[i] = sys.Hrr[i];
        part.visibleEnergy += sys.pointEnergy[i];
      } else {
        sys.pointEnergy[i] = fallbackEnergy[i];
      }
      part.photometricEnergy += sys.pointEnergy[i];

      // Smoothness on the relative difference keeps the prior invariant to the (free) monocular scale.
      double mean = 0;
      for (int j : p.neighbours) mean += rho[j];
      mean = p.neighbours.empty() ? rho[i] : mean / p.neighbours.size();
      const double rel = (rho[i] - mean) / mean;
      const double gauge = (rho[i] - m_gaugeRho) / m_gaugeRho;
      sys.pointPrior[i] = 0.5 * m_settings.smoothWeight * rel * rel + 0.5 * m_settings.gaugeWeight * gauge * gauge;
      part.priorEnergy += sys.pointPrior[i];
      if (withJacobians) {
        sys.Hrr[i] += m_settings.smoothWeight / (mean * mean) + m_settings.gaugeWeight / (m_gaugeRho * m_gaugeRho);
        sys.gr[i] += m_settings.smoothWeight * rel / mean + m_settings.gaugeWeight * gauge / m_gaugeRho;
      }
    }
  });
  for (const Partial& part : partials) {
    sys.Hxx += part.Hxx;
    sys.gx += part.gx;
    sys.numResiduals += part.numResiduals;
    sys.numVisible += part.numVisible;
    sys.numInliers += part.numInliers;
    sys.visibleEnergy += part.visibleEnergy;
    sys.photometricEnergy += part.photometricEnergy;
    sys.energy += part.priorEnergy;
  }
  sys.energy += sys.photometricEnergy;
  return sys;
}

MonoInitializer::System MonoInitializer::optimizeLevel(int level, const ImageLevel& img, State& state,
                                                       int maxIterations) {
  auto& pts = m_levels[level].points;
  const size_t n = pts.size();
  std::vector<double> rho(n), rhoCandidate(n), hr(n), fallback(n);
  for (size_t i = 0; i < n; ++i) {
    rho[i] = pts[i].rho;
    fallback[i] = pts[i].energy;
  }

  System sys = linearize(level, img, state, rho, fallback, true);
  double lambda = 1e-2;
  for (int it = 0; it < maxIterations; ++it) {
    Eigen::Matrix<double, 8, 8> Hs = sys.Hxx;
    Hs.diagonal() *= 1.0 + lambda;
    Hs.diagonal().array() += 1e-9;
    Eigen::Matrix<double, 8, 1> gs = sys.gx;
    for (size_t i = 0; i < n; ++i) {
      hr[i] = sys.Hrr[i] * (1.0 + lambda) + 1e-12;
      Hs.noalias() -= sys.Hxr[i] * sys.Hxr[i].transpose() / hr[i];
      gs.noalias() -= sys.Hxr[i] * (sys.gr[i] / hr[i]);
    }
    const Eigen::Matrix<double, 8, 1> dx = -Hs.ldlt().solve(gs);

    State candidate;
    candidate.T_t_h = Sophus::SE3d::exp(dx.head<6>()) * state.T_t_h;
    candidate.affine = {state.affine.a + dx[6], state.affine.b + dx[7]};
    for (size_t i = 0; i < n; ++i)
      rhoCandidate[i] = std::max(kMinRho, rho[i] - (sys.gr[i] + sys.Hxr[i].dot(dx)) / hr[i]);

    System next = linearize(level, img, candidate, rhoCandidate, sys.pointEnergy, false);
    // A global energy decrease can hide single points jumping to a worse depth; those keep their old value.
    bool reverted = false;
    for (size_t i = 0; i < n; ++i)
      if (next.pointEnergy[i] + next.pointPrior[i] > sys.pointEnergy[i] + sys.pointPrior[i]) {
        rhoCandidate[i] = rho[i];
        reverted = true;
      }
    if (reverted) next = linearize(level, img, candidate, rhoCandidate, sys.pointEnergy, false);
    if (next.energy < sys.energy) {
      state = candidate;
      rho.swap(rhoCandidate);
      sys = linearize(level, img, state, rho, next.pointEnergy, true);
      lambda = std::max(lambda * 0.5, 1e-6);
      if (dx.head<6>().norm() < 1e-6) break;
    } else {
      lambda *= 4.0;
      if (lambda > 1e4) break;
    }
  }

  for (size_t i = 0; i < n; ++i) {
    pts[i].rho = rho[i];
    pts[i].hPhoto = sys.hPhoto[i];
    pts[i].valid = sys.valid[i] != 0;
    pts[i].energy = sys.pointEnergy[i];
  }
  return sys;
}

void MonoInitializer::alignRotation(const ImagePyramid& frame, State& state) const {
  const double k = m_settings.photometric.huberThreshold;
  const double cutoff = m_settings.outlierCutoff;
  auto evaluate = [&](int level, const State& s, Eigen::Matrix<double, 5, 5>* H, Eigen::Matrix<double, 5, 1>* g) {
    HostTargetState hts;
    hts.T_t_h = Sophus::SE3d(s.T_t_h.so3(), Eigen::Vector3d::Zero());
    hts.target = s.affine;
    PatternResidual res;
    Eigen::Matrix<double, 5, 1> J;
    double energy = 0;
    int count = 0;
    if (H) H->setZero();
    if (g) g->setZero();
    for (const auto& p : m_levels[level].points) {
      if (!evaluatePatternResidual(p.pattern, 0.0, hts, m_levels[level].cam, frame.level(level),
                                   m_settings.photometric, res))
        continue;
      for (int q = 0; q < kPatternSize; ++q) {
        const PixelResidual& px = res.pixels[q];
        ++count;
        if (std::abs(px.r) > cutoff) {
          energy += p.pattern.gradientWeights[q] * huberEnergy(cutoff, k);
          continue;
        }
        energy += p.pattern.gradientWeights[q] * huberEnergy(px.r, k);
        if (!H) continue;
        J << px.dPose.tail<3>().transpose(), px.dAffine[2], px.dAffine[3];
        H->noalias() += px.weight * J * J.transpose();
        g->noalias() += px.weight * px.r * J;
      }
    }
    return count > 0 ? energy / count : std::numeric_limits<double>::infinity();
  };

  state.T_t_h.translation().setZero();
  for (int l = static_cast<int>(m_levels.size()) - 1; l >= 0; --l) {
    Eigen::Matrix<double, 5, 5> H;
    Eigen::Matrix<double, 5, 1> g;
    double energy = evaluate(l, state, &H, &g);
    double lambda = 1e-3;
    const auto& its = m_settings.iterations;
    for (int it = 0; it < its[std::min<size_t>(l, its.size() - 1)]; ++it) {
      Eigen::Matrix<double, 5, 5> A = H;
      A.diagonal() *= 1.0 + lambda;
      A.diagonal().array() += 1e-9;
      const Eigen::Matrix<double, 5, 1> d = -A.ldlt().solve(g);
      State candidate;
      candidate.T_t_h = Sophus::SE3d(Sophus::SO3d::exp(d.head<3>()) * state.T_t_h.so3(), Eigen::Vector3d::Zero());
      candidate.affine = {state.affine.a + d[3], state.affine.b + d[4]};
      if (const double e = evaluate(l, candidate, nullptr, nullptr); e < energy) {
        state = candidate;
        energy = evaluate(l, state, &H, &g);
        lambda = std::max(lambda * 0.5, 1e-7);
      } else {
        lambda *= 4.0;
        if (lambda > 1e4) break;
      }
    }
  }
}

double MonoInitializer::translationFlow(const State& state) const {
  const Level& level = m_levels[0];
  std::vector<double> flow;
  Eigen::Vector2d uvFull, uvRot;
  for (const auto& p : level.points) {
    const Eigen::Vector3d r = state.T_t_h.so3() * p.bearing;
    if (!p.valid || !level.cam.project(Eigen::Vector3d(r + p.rho * state.T_t_h.translation()), uvFull) ||
        !level.cam.project(r, uvRot))
      continue;
    flow.push_back((uvFull - uvRot).norm());
  }
  if (flow.empty()) return 0.0;
  std::nth_element(flow.begin(), flow.begin() + flow.size() / 2, flow.end());
  return flow[flow.size() / 2];
}

void MonoInitializer::normalizeScale() {
  double sum = 0;
  int n = 0;
  for (const auto& p : m_levels[0].points)
    if (p.valid) {
      sum += p.rho;
      ++n;
    }
  if (n == 0 || sum <= 0) return;
  const double s = sum / n;
  for (auto& level : m_levels)
    for (auto& p : level.points) {
      p.rho /= s;
      p.rhoStart /= s;
    }
  m_state.T_t_h.translation() *= s;
  m_prevState.T_t_h.translation() *= s;
  m_gaugeRho /= s;
}

std::vector<MonoInitializer::State> MonoInitializer::prescreen(const ImagePyramid& frame,
                                                               const std::vector<State>& starts, int keep) {
  const int top = static_cast<int>(m_levels.size()) - 1;
  const std::vector<Point> snapshot = m_levels[top].points;
  std::vector<std::pair<double, State>> scored;
  const auto& its = m_settings.iterations;
  for (const State& start : starts) {
    m_levels[top].points = snapshot;
    State s = start;
    scored.emplace_back(
        optimizeLevel(top, frame.level(top), s, its[std::min<size_t>(top, its.size() - 1)]).visibleMeanEnergy(), start);
  }
  m_levels[top].points = snapshot;
  std::stable_sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
  std::vector<State> best;
  for (int i = 0; i < keep && i < static_cast<int>(scored.size()); ++i) best.push_back(scored[i].second);
  return best;
}

MonoInitializer::System MonoInitializer::optimizePyramid(const ImagePyramid& frame, State& state) {
  const int top = static_cast<int>(m_levels.size()) - 1;
  System sys;
  for (int l = top; l >= 0; --l) {
    if (l < top)
      for (auto& p : m_levels[l].points) {
        const Point& q = m_levels[l + 1].points[p.parent];
        p.rho *= q.rho / q.rhoStart;
      }
    const auto& its = m_settings.iterations;
    sys = optimizeLevel(l, frame.level(l), state, its[std::min<size_t>(l, its.size() - 1)]);
  }
  return sys;
}

MonoInitResult MonoInitializer::addFrame(const ImagePyramid& frame) {
  MonoInitResult result;
  if (m_levels.empty()) {
    reset(frame);
    result.reset = true;
    result.numFrames = m_numFrames;
    return result;
  }

  const int top = static_cast<int>(m_levels.size()) - 1;
  for (auto& level : m_levels)
    for (auto& p : level.points) p.rhoStart = p.rho;

  std::vector<double> topRho, topEnergy;
  for (const auto& p : m_levels[top].points) {
    topRho.push_back(p.rho);
    topEnergy.push_back(p.energy);
  }
  const State constVel{m_state.T_t_h * m_prevState.T_t_h.inverse() * m_state.T_t_h, m_state.affine};
  State s = linearize(top, frame.level(top), constVel, topRho, topEnergy, false).energy <
                    linearize(top, frame.level(top), m_state, topRho, topEnergy, false).energy
                ? constVel
                : m_state;
  // With flat initial depth, yaw and lateral translation are ambiguous (e.g. a close parked car looks like
  // rotation): on the first frame also try a rotation-only alignment and axis translations.
  std::vector<State> starts = {s};
  if (m_numFrames == 1) {
    alignRotation(frame, starts.emplace_back(s));
    for (double magnitude : m_settings.firstFrameTranslations)
      for (int axis = 0; axis < 3; ++axis)
        for (double sign : {1.0, -1.0}) {
          State& h = starts.emplace_back(s);
          h.T_t_h.translation()[axis] = sign * magnitude / m_gaugeRho;
        }
    starts = prescreen(frame, starts, m_settings.firstFrameCandidates);
  }
  const std::vector<Level> snapshot = m_levels;
  std::vector<Level> bestLevels;
  System sys;
  for (size_t i = 0; i < starts.size(); ++i) {
    if (i > 0) m_levels = snapshot;
    State candidate = starts[i];
    System candidateSys = optimizePyramid(frame, candidate);
    if (i == 0 || candidateSys.visibleMeanEnergy() < sys.visibleMeanEnergy()) {
      s = candidate;
      sys = std::move(candidateSys);
      if (starts.size() > 1) bestLevels = m_levels;
    }
  }
  if (starts.size() > 1) m_levels = std::move(bestLevels);

  for (int l = 1; l <= top; ++l) {
    auto& coarse = m_levels[l].points;
    std::vector<double> sum(coarse.size(), 0.0);
    std::vector<int> count(coarse.size(), 0);
    for (const auto& p : m_levels[l - 1].points) {
      sum[p.parent] += p.rho;
      ++count[p.parent];
    }
    for (size_t j = 0; j < coarse.size(); ++j)
      if (count[j] > 0) coarse[j].rho = sum[j] / count[j];
  }

  m_prevState = m_state;
  m_state = s;
  ++m_numFrames;

  result.inlierRatio = sys.inlierRatio();
  result.rmse = sys.numResiduals > 0 ? std::sqrt(2.0 * sys.photometricEnergy / sys.numResiduals) : 0.0;
  result.translationFlow = translationFlow(s);
  result.visibleRatio = m_levels[0].points.empty() ? 0.0 : double(sys.numVisible) / m_levels[0].points.size();
  if (result.inlierRatio < m_settings.minInlierRatio || result.visibleRatio < m_settings.minVisibleRatio) {
    reset(frame);
    result.reset = true;
    result.numFrames = m_numFrames;
    return result;
  }
  if (m_numFrames >= m_settings.minFrames && result.translationFlow >= m_settings.minTranslationFlow) {
    normalizeScale();
    result.initialized = true;
  }
  result.T_t_h = m_state.T_t_h;
  result.affine = m_state.affine;
  result.numFrames = m_numFrames;
  return result;
}

std::vector<InitPoint> MonoInitializer::points() const {
  std::vector<InitPoint> out;
  if (m_levels.empty()) return out;
  for (const auto& p : m_levels[0].points) {
    const double sigma = p.valid && p.hPhoto > 0 && p.rho > 0 ? 1.0 / (p.rho * std::sqrt(p.hPhoto))
                                                               : std::numeric_limits<double>::infinity();
    out.push_back({p.pattern.uv, p.bearing, p.rho, sigma});
  }
  return out;
}

}  // namespace sdv
