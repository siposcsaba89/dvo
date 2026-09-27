#include <sdv/window_optimizer.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <Eigen/Cholesky>
#include <spdlog/spdlog.h>

namespace sdv {

namespace {

constexpr int kF = 8;

std::vector<int> nonZeroBlocks(const std::vector<Eigen::Matrix<double, kF, 1>>& blocks) {
  std::vector<int> nz;
  for (size_t i = 0; i < blocks.size(); ++i)
    if (!blocks[i].isZero(0)) nz.push_back(static_cast<int>(i));
  return nz;
}

// Eliminates a point's inverse distance: H -= Hfr Hrf / hr, g -= Hfr gr / hr.
void schurPoint(const std::vector<Eigen::Matrix<double, kF, 1>>& Hfr, double gr, double hr, Eigen::MatrixXd& H,
                Eigen::VectorXd& g) {
  const std::vector<int> nz = nonZeroBlocks(Hfr);
  for (int i : nz) {
    g.segment<kF>(kF * i) -= Hfr[i] * (gr / hr);
    for (int j : nz) H.block<kF, kF>(kF * i, kF * j) -= Hfr[i] * Hfr[j].transpose() / hr;
  }
}

}  // namespace

bool evaluateWindowResidual(const PatternPoint& point, double rho, const FrameParams& host,
                            const FrameParams& target, const FrameParams& host0, const FrameParams& target0,
                            const Camera& cam, const ImageLevel& targetImg, const PhotometricSettings& settings,
                            WindowPatternResidual& out) {
  const Sophus::SE3d T_t_h = target.T_c_w * host.T_c_w.inverse();
  const Sophus::SE3d T_t_h0 = target0.T_c_w * host0.T_c_w.inverse();
  const Eigen::Matrix<double, 6, 6> adj = T_t_h0.Adj();
  const double ratio = target.exposure / host.exposure;
  const double scale = ratio * std::exp(target.affine.a - host.affine.a);
  const double scale0 = ratio * std::exp(target0.affine.a - host0.affine.a);

  out.energy = 0;
  for (int k = 0; k < kPatternSize; ++k) {
    Eigen::Vector2d uv, uv0, dUvdRho;
    Eigen::Matrix<double, 2, 6> dUvdPose;
    if (!projectBearing(point.bearings[k], rho, T_t_h, cam, uv) || !cam.isInside(uv.x(), uv.y(), settings.border) ||
        !projectBearing(point.bearings[k], rho, T_t_h0, cam, uv0, &dUvdPose, &dUvdRho))
      return false;

    const Eigen::Vector3f s = targetImg.interpolate(static_cast<float>(uv.x()), static_cast<float>(uv.y()));
    const Eigen::RowVector2d grad(s[1], s[2]);
    const double hostCentered = point.intensities[k] - host.affine.b;
    const double hostCentered0 = point.intensities[k] - host0.affine.b;

    WindowPixelResidual& px = out.pixels[k];
    px.r = (s[0] - target.affine.b) - scale * hostCentered;
    const Eigen::Matrix<double, 1, 6> dRelative = grad * dUvdPose;
    px.dTarget.head<6>() = dRelative;
    px.dHost.head<6>() = -dRelative * adj;
    px.dHost[6] = scale0 * hostCentered0;
    px.dHost[7] = scale0;
    px.dTarget[6] = -scale0 * hostCentered0;
    px.dTarget[7] = -1.0;
    px.dRho = grad * dUvdRho;
    px.weight = point.gradientWeights[k] * huberWeight(px.r, settings.huberThreshold);
    out.energy += point.gradientWeights[k] * huberEnergy(px.r, settings.huberThreshold);
  }
  return true;
}

int WindowPoint::numGood() const {
  return static_cast<int>(std::count_if(residuals.begin(), residuals.end(),
                                        [](const WindowResidual& r) { return r.state == ResidualState::Good; })) +
         (stereoState == ResidualState::Good ? 1 : 0);
}

Eigen::Matrix<double, 8, 1> WindowFrame::delta() const {
  Eigen::Matrix<double, 8, 1> d;
  d.head<6>() = (params.T_c_w * linearization.T_c_w.inverse()).log();
  d[6] = params.affine.a - linearization.affine.a;
  d[7] = params.affine.b - linearization.affine.b;
  return d;
}

WindowOptimizer::WindowOptimizer(const Camera& cam, WindowSettings settings)
    : m_camera(cam), m_settings(std::move(settings)) {}

int WindowOptimizer::frameIndex(int frameId) const {
  for (size_t i = 0; i < m_frames.size(); ++i)
    if (m_frames[i].id == frameId) return static_cast<int>(i);
  throw std::out_of_range("frame not in window");
}

int WindowOptimizer::addFrame(std::shared_ptr<const ImagePyramid> image, const Sophus::SE3d& T_c_w,
                              const AffineBrightness& affine, double exposure) {
  WindowFrame f;
  f.id = m_nextFrameId++;
  f.image = std::move(image);
  f.params = {T_c_w, affine, exposure};
  f.linearization = f.params;

  const Eigen::Index n = kF * static_cast<Eigen::Index>(m_frames.size());
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n + kF, n + kF);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(n + kF);
  H.topLeftCorner(n, n) = m_priorH;
  b.head(n) = m_priorB;
  if (f.id == 0) {
    // Gauge: absolute pose and brightness of the first keyframe.
    f.inPrior = true;
    H.bottomRightCorner<kF, kF>().diagonal().setConstant(m_settings.firstFramePrior);
  }
  m_priorH = std::move(H);
  m_priorB = std::move(b);

  for (auto& p : m_points) p.residuals.push_back({f.id});
  m_frames.push_back(std::move(f));
  return m_frames.back().id;
}

int WindowOptimizer::addPoint(int hostId, const Eigen::Vector2d& uv, double rho) {
  const WindowFrame& host = m_frames[frameIndex(hostId)];
  auto pattern = makePatternPoint(m_camera, host.image->level(0), uv, m_settings.photometric);
  Eigen::Vector3d bearing;
  if (!pattern || !m_camera.unproject(uv, bearing)) return -1;
  WindowPoint p{m_nextPointId++, hostId, *pattern, bearing, rho};
  for (const auto& f : m_frames)
    if (f.id != hostId) p.residuals.push_back({f.id});
  m_points.push_back(std::move(p));
  return m_points.back().id;
}

void WindowOptimizer::setStereo(const Camera& rightCam, const Sophus::SE3d& T_r_l) {
  m_rightCam = rightCam;
  m_T_r_l = T_r_l;
}

void WindowOptimizer::setFrameStereo(int frameId, std::shared_ptr<const ImagePyramid> right,
                                     const AffineBrightness& stereoAffine) {
  WindowFrame& f = m_frames[frameIndex(frameId)];
  f.right = std::move(right);
  f.stereoAffine = stereoAffine;
}

bool WindowOptimizer::evaluateStereo(const WindowPoint& p, double rho, WindowPatternResidual& out) const {
  if (!m_rightCam) return false;
  const WindowFrame& host = m_frames[frameIndex(p.host)];
  if (!host.right) return false;
  // Host at the origin with neutral brightness, so the pattern's raw intensities map through the stereo affine.
  const FrameParams left{Sophus::SE3d(), {}, 1.0};
  const FrameParams right{m_T_r_l, host.stereoAffine, 1.0};
  if (!evaluateWindowResidual(p.pattern, rho, left, right, left, right, *m_rightCam, host.right->level(0),
                              m_settings.photometric, out))
    return false;
  for (auto& px : out.pixels) px.weight *= m_settings.stereoWeight;
  out.energy *= m_settings.stereoWeight;
  return true;
}

void WindowOptimizer::removePoint(int pointId) {
  std::erase_if(m_points, [&](const WindowPoint& p) { return p.id == pointId; });
}

WindowOptimizer::System WindowOptimizer::linearize(const std::vector<size_t>& pointIndices) const {
  const size_t nf = m_frames.size();
  System sys;
  sys.H = Eigen::MatrixXd::Zero(kF * nf, kF * nf);
  sys.g = Eigen::VectorXd::Zero(kF * nf);

  std::vector<size_t> indices = pointIndices;
  if (indices.empty())
    for (size_t i = 0; i < m_points.size(); ++i) indices.push_back(i);
  sys.points.resize(indices.size());

  WindowPatternResidual res;
  for (size_t pi = 0; pi < indices.size(); ++pi) {
    const WindowPoint& p = m_points[indices[pi]];
    PointBlock& pb = sys.points[pi];
    pb.Hfr.assign(nf, Eigen::Matrix<double, kF, 1>::Zero());
    const int h = frameIndex(p.host);
    const WindowFrame& hf = m_frames[h];
    for (const auto& r : p.residuals) {
      if (r.state != ResidualState::Good) continue;
      const int t = frameIndex(r.target);
      const WindowFrame& tf = m_frames[t];
      if (!evaluateWindowResidual(p.pattern, p.rho, hf.params, tf.params, hf.linearization, tf.linearization,
                                  m_camera, tf.image->level(0), m_settings.photometric, res))
        continue;
      Eigen::Matrix<double, kF, kF> Hhh = Eigen::Matrix<double, kF, kF>::Zero(), Hht = Hhh, Htt = Hhh;
      Eigen::Matrix<double, kF, 1> gh = Eigen::Matrix<double, kF, 1>::Zero(), gt = gh, hrh = gh, hrt = gh;
      for (const auto& px : res.pixels) {
        const Eigen::Matrix<double, kF, 1> Jh = px.dHost.transpose(), Jt = px.dTarget.transpose();
        Hhh.noalias() += px.weight * Jh * Jh.transpose();
        Hht.noalias() += px.weight * Jh * Jt.transpose();
        Htt.noalias() += px.weight * Jt * Jt.transpose();
        gh.noalias() += px.weight * px.r * Jh;
        gt.noalias() += px.weight * px.r * Jt;
        hrh.noalias() += px.weight * px.dRho * Jh;
        hrt.noalias() += px.weight * px.dRho * Jt;
        pb.Hrr += px.weight * px.dRho * px.dRho;
        pb.gr += px.weight * px.r * px.dRho;
      }
      sys.H.block<kF, kF>(kF * h, kF * h) += Hhh;
      sys.H.block<kF, kF>(kF * h, kF * t) += Hht;
      sys.H.block<kF, kF>(kF * t, kF * h) += Hht.transpose();
      sys.H.block<kF, kF>(kF * t, kF * t) += Htt;
      sys.g.segment<kF>(kF * h) += gh;
      sys.g.segment<kF>(kF * t) += gt;
      pb.Hfr[h] += hrh;
      pb.Hfr[t] += hrt;
    }
    if (p.stereoState == ResidualState::Good && evaluateStereo(p, p.rho, res))
      for (const auto& px : res.pixels) {
        pb.Hrr += px.weight * px.dRho * px.dRho;
        pb.gr += px.weight * px.r * px.dRho;
      }
  }
  return sys;
}

void WindowOptimizer::classifyResiduals() {
  const double k = m_settings.photometric.huberThreshold;
  const double thresholdEnergy = huberEnergy(m_settings.outlierThreshold, k);
  WindowPatternResidual res;
  for (auto& p : m_points) {
    const WindowFrame& hf = m_frames[frameIndex(p.host)];
    double maxEnergy = 0;
    for (float w : p.pattern.gradientWeights) maxEnergy += w * thresholdEnergy;
    for (auto& r : p.residuals) {
      const WindowFrame& tf = m_frames[frameIndex(r.target)];
      if (!evaluateWindowResidual(p.pattern, p.rho, hf.params, tf.params, hf.params, tf.params, m_camera,
                                  tf.image->level(0), m_settings.photometric, res)) {
        r.state = ResidualState::OutOfBounds;
        continue;
      }
      r.energy = res.energy;
      r.state = res.energy > maxEnergy ? ResidualState::Outlier : ResidualState::Good;
    }
    if (!evaluateStereo(p, p.rho, res)) {
      p.stereoState = ResidualState::OutOfBounds;
      continue;
    }
    p.stereoEnergy = res.energy;
    p.stereoState = res.energy > maxEnergy * m_settings.stereoWeight ? ResidualState::Outlier : ResidualState::Good;
  }
}

double WindowOptimizer::priorEnergy() const {
  Eigen::VectorXd delta(kF * m_frames.size());
  double e = 0;
  for (size_t i = 0; i < m_frames.size(); ++i) {
    delta.segment<kF>(kF * i) = m_frames[i].delta();
    const AffineBrightness& a = m_frames[i].params.affine;
    e += 0.5 * m_settings.affinePriorA * a.a * a.a + 0.5 * m_settings.affinePriorB * a.b * a.b;
  }
  return e + 0.5 * delta.dot(m_priorH * delta) + m_priorB.dot(delta);
}

void WindowOptimizer::addPriors(System& sys) const {
  Eigen::VectorXd delta(kF * m_frames.size());
  for (size_t i = 0; i < m_frames.size(); ++i) {
    delta.segment<kF>(kF * i) = m_frames[i].delta();
    const AffineBrightness& a = m_frames[i].params.affine;
    sys.H(kF * i + 6, kF * i + 6) += m_settings.affinePriorA;
    sys.H(kF * i + 7, kF * i + 7) += m_settings.affinePriorB;
    sys.g[kF * i + 6] += m_settings.affinePriorA * a.a;
    sys.g[kF * i + 7] += m_settings.affinePriorB * a.b;
  }
  sys.H += m_priorH;
  sys.g += m_priorH * delta + m_priorB;
}

double WindowOptimizer::pointEnergy(const WindowPoint& p, double rho) const {
  double e = 0;
  WindowPatternResidual res;
  const WindowFrame& hf = m_frames[frameIndex(p.host)];
  for (const auto& r : p.residuals) {
    if (r.state != ResidualState::Good) continue;
    const WindowFrame& tf = m_frames[frameIndex(r.target)];
    // Leaving the image keeps the classification energy, so it is neither rewarded nor penalised.
    e += evaluateWindowResidual(p.pattern, rho, hf.params, tf.params, hf.params, tf.params, m_camera,
                                tf.image->level(0), m_settings.photometric, res)
             ? res.energy
             : r.energy;
  }
  if (p.stereoState == ResidualState::Good) e += evaluateStereo(p, rho, res) ? res.energy : p.stereoEnergy;
  return e;
}

double WindowOptimizer::energy() const {
  double e = 0;
  for (const auto& p : m_points) e += pointEnergy(p, p.rho);
  return e + priorEnergy();
}

WindowOptimizationResult WindowOptimizer::optimize(int maxIterations) {
  WindowOptimizationResult result;
  if (m_frames.size() < 2) return result;
  if (maxIterations < 0) maxIterations = m_settings.maxIterations;

  classifyResiduals();
  for (const auto& p : m_points)
    for (const auto& r : p.residuals) {
      result.numGood += r.state == ResidualState::Good;
      result.numOutliers += r.state == ResidualState::Outlier;
      result.numOutOfBounds += r.state == ResidualState::OutOfBounds;
    }

  double current = energy();
  result.initialEnergy = current;
  double lambda = 1e-4;
  std::vector<FrameParams> backupFrames(m_frames.size());
  std::vector<double> backupRho(m_points.size());
  for (int it = 0; it < maxIterations; ++it) {
    for (auto& f : m_frames)
      if (!f.inPrior) f.linearization = f.params;
    System sys = linearize({});
    addPriors(sys);
    for (size_t i = 0; i < m_points.size(); ++i) m_points[i].hRho = sys.points[i].Hrr;

    Eigen::MatrixXd A = sys.H;
    A.diagonal() *= 1.0 + lambda;
    A.diagonal().array() += 1e-8;
    Eigen::VectorXd b = sys.g;
    std::vector<double> hr(m_points.size());
    for (size_t i = 0; i < m_points.size(); ++i) {
      hr[i] = sys.points[i].Hrr * (1.0 + lambda) + 1e-12;
      schurPoint(sys.points[i].Hfr, sys.points[i].gr, hr[i], A, b);
    }
    const Eigen::VectorXd dx = -A.ldlt().solve(b);

    for (size_t i = 0; i < m_frames.size(); ++i) {
      backupFrames[i] = m_frames[i].params;
      FrameParams& fp = m_frames[i].params;
      fp.T_c_w = Sophus::SE3d::exp(dx.segment<6>(kF * i)) * fp.T_c_w;
      fp.affine.a += dx[kF * i + 6];
      fp.affine.b += dx[kF * i + 7];
    }
    for (size_t i = 0; i < m_points.size(); ++i) {
      backupRho[i] = m_points[i].rho;
      double d = sys.points[i].gr;
      for (size_t f = 0; f < m_frames.size(); ++f) d += sys.points[i].Hfr[f].dot(dx.segment<kF>(kF * f));
      const double rho = m_points[i].rho - d / hr[i];
      m_points[i].rho = rho > 0 ? rho : 0.5 * m_points[i].rho;
    }

    // Given the frames, points are independent: a point keeps its depth step (or half of it) only if that lowers
    // its own energy. The photometric error is far from linear in depth; without this about half of the points
    // overshoot in every step and one bad point rejects the whole step, so the window never converges (KITTI 00).
    ++result.iterations;
    double next = priorEnergy();
    int reverted = 0;
    for (size_t i = 0; i < m_points.size(); ++i) {
      WindowPoint& p = m_points[i];
      double e = pointEnergy(p, p.rho);
      if (const double eOld = pointEnergy(p, backupRho[i]); eOld < e) {
        const double half = 0.5 * (p.rho + backupRho[i]);
        const double eHalf = pointEnergy(p, half);
        if (eHalf < eOld) {
          p.rho = half;
          e = eHalf;
        } else {
          p.rho = backupRho[i];
          e = eOld;
          ++reverted;
        }
      }
      next += e;
    }
    spdlog::trace("window BA iteration {}: lambda {:.1e}, energy {:.1f} -> {:.1f}, {} of {} depth steps reverted", it,
                  lambda, current, next, reverted, m_points.size());

    if (next < current) {
      current = next;
      lambda = std::max(lambda * 0.5, 1e-7);
      if (dx.lpNorm<Eigen::Infinity>() < 1e-7) break;
    } else {
      for (size_t i = 0; i < m_frames.size(); ++i) m_frames[i].params = backupFrames[i];
      for (size_t i = 0; i < m_points.size(); ++i) m_points[i].rho = backupRho[i];
      lambda *= 4.0;
      if (lambda > 1e4) break;
    }
  }
  result.finalEnergy = current;
  return result;
}

void WindowOptimizer::marginalizeFrame(int frameId) {
  const int k = frameIndex(frameId);
  const Eigen::Index nf = static_cast<Eigen::Index>(m_frames.size());
  for (auto& f : m_frames)
    if (!f.inPrior) f.linearization = f.params;
  classifyResiduals();

  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(kF * nf, kF * nf);
  Eigen::VectorXd g = Eigen::VectorXd::Zero(kF * nf);
  for (size_t i = 0; i < m_points.size(); ++i) {
    if (m_points[i].host != frameId) continue;
    System sys = linearize({i});
    const PointBlock& pb = sys.points[0];
    // Points with too little depth information are dropped rather than marginalised.
    if (pb.Hrr < m_settings.minRhoHessian) continue;
    schurPoint(pb.Hfr, pb.gr, pb.Hrr, sys.H, sys.g);
    H += sys.H;
    g += sys.g;
  }
  Eigen::VectorXd delta(kF * nf);
  for (Eigen::Index i = 0; i < nf; ++i) delta.segment<kF>(kF * i) = m_frames[i].delta();
  // The linearisation is around the current state; the prior is expressed in delta coordinates.
  m_priorH += H;
  m_priorB += g - H * delta;
  for (Eigen::Index i = 0; i < nf; ++i)
    if (!m_frames[i].inPrior && !m_priorH.block<kF, kF>(kF * i, kF * i).isZero(0)) m_frames[i].inPrior = true;

  std::vector<Eigen::Index> keep;
  for (Eigen::Index i = 0; i < kF * nf; ++i)
    if (i / kF != k) keep.push_back(i);
  const Eigen::Index nk = static_cast<Eigen::Index>(keep.size());
  Eigen::MatrixXd Hrr(nk, nk), Hrk(nk, kF);
  Eigen::VectorXd br(nk);
  for (Eigen::Index i = 0; i < nk; ++i) {
    br[i] = m_priorB[keep[i]];
    Hrk.row(i) = m_priorH.block<1, kF>(keep[i], kF * k);
    for (Eigen::Index j = 0; j < nk; ++j) Hrr(i, j) = m_priorH(keep[i], keep[j]);
  }
  Eigen::Matrix<double, kF, kF> Hkk = m_priorH.block<kF, kF>(kF * k, kF * k);
  Hkk.diagonal().array() += 1e-9 * std::max(Hkk.trace(), 1.0);
  const Eigen::Matrix<double, kF, kF> HkkInv = Hkk.ldlt().solve(Eigen::Matrix<double, kF, kF>::Identity());
  const Eigen::MatrixXd reduced = Hrr - Hrk * HkkInv * Hrk.transpose();
  const Eigen::Matrix<double, kF, 1> bk = m_priorB.segment<kF>(kF * k);
  m_priorB = br - Hrk * (HkkInv * bk);
  m_priorH = 0.5 * (reduced + reduced.transpose());

  std::erase_if(m_points, [&](const WindowPoint& p) { return p.host == frameId; });
  for (auto& p : m_points)
    std::erase_if(p.residuals, [&](const WindowResidual& r) { return r.target == frameId; });
  m_frames.erase(m_frames.begin() + k);
}


void WindowOptimizer::logStereoDepthBias() const {
  if (!m_rightCam) return;
  constexpr int kSteps = 61;
  constexpr double kRange = 0.15;  // searched log inverse depth range around the start value
  const double inf = std::numeric_limits<double>::infinity();
  auto argminLog = [&](auto&& energyAt, double rho) {
    double best = inf, bestL = 0;
    for (int s = 0; s < kSteps; ++s) {
      const double l = -kRange + 2 * kRange * s / (kSteps - 1);
      if (const double e = energyAt(rho * std::exp(l)); e < best) best = e, bestL = l;
    }
    return std::abs(bestL) < 0.99 * kRange ? std::optional(bestL) : std::nullopt;
  };

  double sum[3] = {};
  int count[3] = {};
  WindowPatternResidual res;
  for (const auto& p : m_points) {
    if (p.stereoState != ResidualState::Good || p.rho <= 0) continue;
    const auto ls = argminLog([&](double r) { return evaluateStereo(p, r, res) ? res.energy : inf; }, p.rho);
    if (!ls) continue;
    const double rhoStereo = p.rho * std::exp(*ls);
    const WindowFrame& hf = m_frames[frameIndex(p.host)];
    const double radius = std::hypot(p.pattern.uv.x() - m_camera.cx, p.pattern.uv.y() - m_camera.cy) /
                          std::hypot(m_camera.width / 2.0, m_camera.height / 2.0);
    const int bin = std::min(2, static_cast<int>(radius * 3));
    for (const auto& r : p.residuals) {
      if (r.state != ResidualState::Good) continue;
      const WindowFrame& tf = m_frames[frameIndex(r.target)];
      const auto lt = argminLog(
          [&](double rho) {
            return evaluateWindowResidual(p.pattern, rho, hf.params, tf.params, hf.params, tf.params, m_camera,
                                          tf.image->level(0), m_settings.photometric, res)
                       ? res.energy
                       : inf;
          },
          rhoStereo);
      if (!lt) continue;
      sum[bin] += *lt;
      ++count[bin];
    }
  }
  auto mean = [&](int b) { return count[b] ? 100.0 * sum[b] / count[b] : 0.0; };
  spdlog::info("temporal vs stereo inverse depth by image radius (inner/middle/outer third): {:+.2f} {:+.2f} {:+.2f} % "
               "({} {} {} residuals)",
               mean(0), mean(1), mean(2), count[0], count[1], count[2]);
}

}  // namespace sdv
