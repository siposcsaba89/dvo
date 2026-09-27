#include <sdv/window_optimizer.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>

#include <Eigen/Cholesky>
#include <spdlog/spdlog.h>

#include <sdv/parallel.h>

namespace sdv {

WindowPairState makeWindowPairState(const FrameParams& host, const FrameParams& target, const FrameParams& host0,
                                    const FrameParams& target0) {
  return makeWindowPairState(host, target, host0, target0, Sophus::SE3d(), Sophus::SE3d(), false);
}

WindowPairState makeWindowPairState(const FrameParams& host, const FrameParams& target, const FrameParams& host0,
                                    const FrameParams& target0, const Sophus::SE3d& T_ch_b, const Sophus::SE3d& T_ct_b,
                                    bool sameFrame) {
  WindowPairState s;
  // Cameras of one keyframe: the relative pose is the extrinsic one, exactly and independent of linearisation.
  const Sophus::SE3d T_t_h = sameFrame ? T_ct_b * T_ch_b.inverse() : target.T_c_w * host.T_c_w.inverse();
  const Sophus::SE3d T_t_h0 = sameFrame ? T_t_h : target0.T_c_w * host0.T_c_w.inverse();
  s.R = T_t_h.rotationMatrix();
  s.t = T_t_h.translation();
  s.R0 = T_t_h0.rotationMatrix();
  s.t0 = T_t_h0.translation();
  s.adjHost0 = (T_t_h0 * T_ch_b).Adj();
  s.adjTarget = T_ct_b.Adj();
  s.identityTarget = T_ct_b.params() == Sophus::SE3d().params();
  s.sameFrame = sameFrame;
  const double ratio = target.exposure / host.exposure;
  s.scale = ratio * std::exp(target.affine.a - host.affine.a);
  s.scale0 = ratio * std::exp(target0.affine.a - host0.affine.a);
  s.hostB = host.affine.b;
  s.hostB0 = host0.affine.b;
  s.targetB = target.affine.b;
  s.linearizedAtCurrent = T_t_h.params() == T_t_h0.params();
  return s;
}

bool evaluateWindowResidual(const PatternPoint& point, double rho, const FrameParams& host,
                            const FrameParams& target, const FrameParams& host0, const FrameParams& target0,
                            const Camera& cam, const ImageLevel& targetImg, const PhotometricSettings& settings,
                            WindowPatternResidual& out) {
  return evaluateWindowResidual(point, rho, makeWindowPairState(host, target, host0, target0), cam, targetImg,
                                settings, out);
}

bool evaluateWindowResidual(const PatternPoint& point, double rho, const WindowPairState& pair, const Camera& cam,
                            const ImageLevel& targetImg, const PhotometricSettings& settings,
                            WindowPatternResidual& out) {
  out.energy = 0;
  for (int k = 0; k < kPatternSize; ++k) {
    Eigen::Vector2d uv, uv0, dUvdRho;
    Eigen::Matrix<double, 2, 6> dUvdPose;
    if (pair.linearizedAtCurrent) {
      if (!projectBearing(point.bearings[k], rho, pair.R, pair.t, cam, uv, &dUvdPose, &dUvdRho) ||
          !cam.isInside(uv.x(), uv.y(), settings.border))
        return false;
    } else if (!projectBearing(point.bearings[k], rho, pair.R, pair.t, cam, uv) ||
               !cam.isInside(uv.x(), uv.y(), settings.border) ||
               !projectBearing(point.bearings[k], rho, pair.R0, pair.t0, cam, uv0, &dUvdPose, &dUvdRho)) {
      return false;
    }

    const Eigen::Vector3f s = targetImg.interpolate(static_cast<float>(uv.x()), static_cast<float>(uv.y()));
    const Eigen::RowVector2d grad(s[1], s[2]);
    const double hostCentered = point.intensities[k] - pair.hostB;
    const double hostCentered0 = point.intensities[k] - pair.hostB0;

    WindowPixelResidual& px = out.pixels[k];
    px.r = (s[0] - pair.targetB) - pair.scale * hostCentered;
    if (pair.sameFrame) {
      px.dTarget.head<6>().setZero();
      px.dHost.head<6>().setZero();
    } else {
      const Eigen::Matrix<double, 1, 6> dRelative = grad * dUvdPose;
      px.dTarget.head<6>() = pair.identityTarget ? dRelative : Eigen::Matrix<double, 1, 6>(dRelative * pair.adjTarget);
      px.dHost.head<6>() = -dRelative * pair.adjHost0;
    }
    px.dHost[6] = pair.scale0 * hostCentered0;
    px.dHost[7] = pair.scale0;
    px.dTarget[6] = -pair.scale0 * hostCentered0;
    px.dTarget[7] = -1.0;
    px.dRho = grad * dUvdRho;
    px.weight = point.gradientWeights[k] * huberWeight(px.r, settings.huberThreshold);
    out.energy += point.gradientWeights[k] * huberEnergy(px.r, settings.huberThreshold);
  }
  return true;
}

bool windowResidualEnergy(const PatternPoint& point, double rho, const WindowPairState& pair, const Camera& cam,
                          const ImageLevel& targetImg, const PhotometricSettings& settings, double& energy) {
  energy = 0;
  for (int k = 0; k < kPatternSize; ++k) {
    Eigen::Vector2d uv;
    if (!projectBearing(point.bearings[k], rho, pair.R, pair.t, cam, uv) || !cam.isInside(uv.x(), uv.y(), settings.border))
      return false;
    const double r = (targetImg.interpolateIntensity(static_cast<float>(uv.x()), static_cast<float>(uv.y())) -
                      pair.targetB) -
                     pair.scale * (point.intensities[k] - pair.hostB);
    energy += point.gradientWeights[k] * huberEnergy(r, settings.huberThreshold);
  }
  return true;
}

int WindowPoint::numGood() const {
  return static_cast<int>(std::count_if(residuals.begin(), residuals.end(),
                                        [](const WindowResidual& r) { return r.state == ResidualState::Good; }));
}

Eigen::VectorXd WindowFrame::delta() const {
  const int nc = static_cast<int>(params.affine.size());
  Eigen::VectorXd d(6 + 2 * nc);
  d.head<6>() = (params.T_b_w * linearization.T_b_w.inverse()).log();
  for (int c = 0; c < nc; ++c) {
    d[6 + 2 * c] = params.affine[c].a - linearization.affine[c].a;
    d[7 + 2 * c] = params.affine[c].b - linearization.affine[c].b;
  }
  return d;
}

WindowOptimizer::WindowOptimizer(const Rig& rig, WindowSettings settings) : m_rig(rig), m_settings(std::move(settings)) {
  if (m_rig.size() == 0 || m_rig.T_c_b.size() != m_rig.cameras.size())
    throw std::invalid_argument("rig needs one extrinsic per camera");
}

int WindowOptimizer::frameIndex(int frameId) const {
  for (size_t i = 0; i < m_frames.size(); ++i)
    if (m_frames[i].id == frameId) return static_cast<int>(i);
  throw std::out_of_range("frame not in window");
}

std::array<int, 8> WindowOptimizer::parameterIndices(int f, int c) const {
  const int base = frameDim() * f;
  return {base, base + 1, base + 2, base + 3, base + 4, base + 5, base + 6 + 2 * c, base + 7 + 2 * c};
}

int WindowOptimizer::addFrame(std::vector<std::shared_ptr<const ImagePyramid>> images, const Sophus::SE3d& T_b_w,
                              std::vector<AffineBrightness> affine, std::vector<double> exposure) {
  const int nc = m_rig.size();
  if (static_cast<int>(images.size()) != nc) throw std::invalid_argument("one image per rig camera required");
  if (affine.empty()) affine.resize(nc);
  if (exposure.empty()) exposure.assign(nc, 1.0);
  WindowFrame f;
  f.id = m_nextFrameId++;
  f.images = std::move(images);
  f.params = {T_b_w, std::move(affine), std::move(exposure)};
  f.linearization = f.params;

  const int d = frameDim();
  const Eigen::Index n = d * static_cast<Eigen::Index>(m_frames.size());
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n + d, n + d);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(n + d);
  H.topLeftCorner(n, n) = m_priorH;
  b.head(n) = m_priorB;
  if (f.id == 0) {
    // Gauge: absolute pose of the first keyframe and brightness of its first camera.
    f.inPrior = true;
    for (int i = 0; i < 8; ++i) H(n + i, n + i) = m_settings.firstFramePrior;
  }
  m_priorH = std::move(H);
  m_priorB = std::move(b);

  for (auto& p : m_points)
    for (int c = 0; c < nc; ++c) p.residuals.push_back({f.id, c});
  m_frames.push_back(std::move(f));
  return m_frames.back().id;
}

int WindowOptimizer::addPoint(int hostId, const Eigen::Vector2d& uv, double rho, int hostCam) {
  const WindowFrame& host = m_frames[frameIndex(hostId)];
  const Camera& cam = m_rig.cameras[hostCam];
  auto pattern = makePatternPoint(cam, host.images[hostCam]->level(0), uv, m_settings.photometric);
  Eigen::Vector3d bearing;
  if (!pattern || !cam.unproject(uv, bearing)) return -1;
  WindowPoint p{m_nextPointId++, hostId, hostCam, *pattern, bearing, rho};
  for (const auto& f : m_frames)
    for (int c = 0; c < m_rig.size(); ++c)
      if (f.id != hostId || c != hostCam) p.residuals.push_back({f.id, c});
  m_points.push_back(std::move(p));
  return m_points.back().id;
}

void WindowOptimizer::removePoint(int pointId) {
  std::erase_if(m_points, [&](const WindowPoint& p) { return p.id == pointId; });
}

WindowOptimizer::Pairs WindowOptimizer::makePairs() const {
  Pairs pairs;
  const int nf = static_cast<int>(m_frames.size()), nc = m_rig.size();
  pairs.numFrames = nf;
  pairs.numCams = nc;
  pairs.states.resize(static_cast<size_t>(nf * nc) * (nf * nc));
  for (int h = 0; h < nf; ++h)
    for (int hc = 0; hc < nc; ++hc) {
      const FrameParams host = m_frames[h].params.camera(m_rig, hc);
      const FrameParams host0 = m_frames[h].linearization.camera(m_rig, hc);
      for (int t = 0; t < nf; ++t)
        for (int tc = 0; tc < nc; ++tc) {
          if (h == t && hc == tc) continue;
          pairs.states[((h * nc + hc) * nf + t) * nc + tc] = makeWindowPairState(
              host, m_frames[t].params.camera(m_rig, tc), host0, m_frames[t].linearization.camera(m_rig, tc),
              m_rig.T_c_b[hc], m_rig.T_c_b[tc], h == t);
        }
    }
  return pairs;
}

WindowOptimizer::System WindowOptimizer::linearize(const Pairs& pairs, const std::vector<size_t>& pointIndices) const {
  const Eigen::Index n = frameDim() * static_cast<Eigen::Index>(m_frames.size());
  std::vector<size_t> indices = pointIndices;
  if (indices.empty())
    for (size_t i = 0; i < m_points.size(); ++i) indices.push_back(i);
  System sys;
  sys.points.resize(indices.size());

  std::vector<Eigen::MatrixXd> chunkH(kParallelChunks);
  std::vector<Eigen::VectorXd> chunkG(kParallelChunks);
  parallelChunks(indices.size(), [&](size_t c, size_t begin, size_t end) {
    Eigen::MatrixXd& H = chunkH[c] = Eigen::MatrixXd::Zero(n, n);
    Eigen::VectorXd& g = chunkG[c] = Eigen::VectorXd::Zero(n);
    WindowPatternResidual res;
    for (size_t pi = begin; pi < end; ++pi) {
      const WindowPoint& p = m_points[indices[pi]];
      PointBlock& pb = sys.points[pi];
      pb.Hfr = Eigen::VectorXd::Zero(n);
      const int h = frameIndex(p.host);
      const std::array<int, 8> ih = parameterIndices(h, p.hostCam);
      for (const auto& r : p.residuals) {
        if (r.state != ResidualState::Good) continue;
        const int t = frameIndex(r.target);
        if (!evaluateWindowResidual(p.pattern, p.rho, pairs.at(h, p.hostCam, t, r.targetCam),
                                    m_rig.cameras[r.targetCam], m_frames[t].images[r.targetCam]->level(0),
                                    m_settings.photometric, res))
          continue;
        const double weightScale = t == h ? m_settings.stereoWeight : 1.0;
        Eigen::Matrix<double, 8, 8> Hhh = Eigen::Matrix<double, 8, 8>::Zero(), Hht = Hhh, Htt = Hhh;
        Eigen::Matrix<double, 8, 1> gh = Eigen::Matrix<double, 8, 1>::Zero(), gt = gh, hrh = gh, hrt = gh;
        for (const auto& px : res.pixels) {
          const double w = weightScale * px.weight;
          const Eigen::Matrix<double, 8, 1> Jh = px.dHost.transpose(), Jt = px.dTarget.transpose();
          Hhh.noalias() += w * Jh * Jh.transpose();
          Hht.noalias() += w * Jh * Jt.transpose();
          Htt.noalias() += w * Jt * Jt.transpose();
          gh.noalias() += w * px.r * Jh;
          gt.noalias() += w * px.r * Jt;
          hrh.noalias() += w * px.dRho * Jh;
          hrt.noalias() += w * px.dRho * Jt;
          pb.Hrr += w * px.dRho * px.dRho;
          pb.gr += w * px.r * px.dRho;
        }
        const std::array<int, 8> it = parameterIndices(t, r.targetCam);
        for (int a = 0; a < 8; ++a) {
          for (int b = 0; b < 8; ++b) {
            H(ih[a], ih[b]) += Hhh(a, b);
            H(ih[a], it[b]) += Hht(a, b);
            H(it[a], ih[b]) += Hht(b, a);
            H(it[a], it[b]) += Htt(a, b);
          }
          g[ih[a]] += gh[a];
          g[it[a]] += gt[a];
          pb.Hfr[ih[a]] += hrh[a];
          pb.Hfr[it[a]] += hrt[a];
        }
        for (int f : {h, t})
          if (std::find(pb.frames.begin(), pb.frames.end(), f) == pb.frames.end()) pb.frames.push_back(f);
      }
    }
  });
  sys.H = Eigen::MatrixXd::Zero(n, n);
  sys.g = Eigen::VectorXd::Zero(n);
  for (size_t c = 0; c < kParallelChunks; ++c) {
    sys.H += chunkH[c];
    sys.g += chunkG[c];
  }
  return sys;
}

// Eliminates a point's inverse distance: H -= Hfr Hrf / hr, g -= Hfr gr / hr.
void WindowOptimizer::schurPoint(const PointBlock& pb, double hr, Eigen::MatrixXd& H, Eigen::VectorXd& g) const {
  const int d = frameDim();
  for (int i : pb.frames) {
    const auto hi = pb.Hfr.segment(d * i, d);
    g.segment(d * i, d) -= hi * (pb.gr / hr);
    for (int j : pb.frames) H.block(d * i, d * j, d, d).noalias() -= hi * pb.Hfr.segment(d * j, d).transpose() / hr;
  }
}

bool WindowOptimizer::residualEnergy(const Pairs& pairs, const WindowPoint& p, const WindowResidual& r, double rho,
                                     double& energy) const {
  const int h = frameIndex(p.host), t = frameIndex(r.target);
  if (!windowResidualEnergy(p.pattern, rho, pairs.at(h, p.hostCam, t, r.targetCam), m_rig.cameras[r.targetCam],
                            m_frames[t].images[r.targetCam]->level(0), m_settings.photometric, energy))
    return false;
  if (t == h) energy *= m_settings.stereoWeight;
  return true;
}

void WindowOptimizer::classifyResiduals() {
  const Pairs pairs = makePairs();
  const double thresholdEnergy = huberEnergy(m_settings.outlierThreshold, m_settings.photometric.huberThreshold);
  parallelChunks(m_points.size(), [&](size_t, size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i) {
      WindowPoint& p = m_points[i];
      double maxEnergy = 0;
      for (float w : p.pattern.gradientWeights) maxEnergy += w * thresholdEnergy;
      for (auto& r : p.residuals) {
        double e;
        if (!residualEnergy(pairs, p, r, p.rho, e)) {
          r.state = ResidualState::OutOfBounds;
          continue;
        }
        r.energy = e;
        const double limit = r.target == p.host ? maxEnergy * m_settings.stereoWeight : maxEnergy;
        r.state = e > limit ? ResidualState::Outlier : ResidualState::Good;
      }
    }
  });
}

double WindowOptimizer::priorEnergy() const {
  const int d = frameDim();
  Eigen::VectorXd delta(d * m_frames.size());
  double e = 0;
  for (size_t i = 0; i < m_frames.size(); ++i) {
    delta.segment(d * i, d) = m_frames[i].delta();
    for (const AffineBrightness& a : m_frames[i].params.affine)
      e += 0.5 * m_settings.affinePriorA * a.a * a.a + 0.5 * m_settings.affinePriorB * a.b * a.b;
  }
  return e + 0.5 * delta.dot(m_priorH * delta) + m_priorB.dot(delta);
}

void WindowOptimizer::addPriors(System& sys) const {
  const int d = frameDim();
  Eigen::VectorXd delta(d * m_frames.size());
  for (size_t i = 0; i < m_frames.size(); ++i) {
    delta.segment(d * i, d) = m_frames[i].delta();
    for (int c = 0; c < m_rig.size(); ++c) {
      const AffineBrightness& a = m_frames[i].params.affine[c];
      const size_t ia = d * i + 6 + 2 * c;
      sys.H(ia, ia) += m_settings.affinePriorA;
      sys.H(ia + 1, ia + 1) += m_settings.affinePriorB;
      sys.g[ia] += m_settings.affinePriorA * a.a;
      sys.g[ia + 1] += m_settings.affinePriorB * a.b;
    }
  }
  sys.H += m_priorH;
  sys.g += m_priorH * delta + m_priorB;
}

double WindowOptimizer::pointEnergy(const Pairs& pairs, const WindowPoint& p, double rho) const {
  double sum = 0;
  for (const auto& r : p.residuals) {
    if (r.state != ResidualState::Good) continue;
    // Leaving the image keeps the classification energy, so it is neither rewarded nor penalised.
    double e;
    sum += residualEnergy(pairs, p, r, rho, e) ? e : r.energy;
  }
  return sum;
}

double WindowOptimizer::energy(const Pairs& pairs) const {
  std::vector<double> chunkEnergy(kParallelChunks, 0.0);
  parallelChunks(m_points.size(), [&](size_t c, size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i) chunkEnergy[c] += pointEnergy(pairs, m_points[i], m_points[i].rho);
  });
  double e = priorEnergy();
  for (double c : chunkEnergy) e += c;
  return e;
}

WindowOptimizationResult WindowOptimizer::optimize(int maxIterations) {
  WindowOptimizationResult result;
  if (m_frames.size() < 2) return result;
  if (maxIterations < 0) maxIterations = m_settings.maxIterations;
  const int d = frameDim();

  classifyResiduals();
  for (const auto& p : m_points)
    for (const auto& r : p.residuals) {
      result.numGood += r.state == ResidualState::Good;
      result.numOutliers += r.state == ResidualState::Outlier;
      result.numOutOfBounds += r.state == ResidualState::OutOfBounds;
    }

  double current = energy(makePairs());
  result.initialEnergy = current;
  double lambda = 1e-4;
  std::vector<KeyframeParams> backupFrames(m_frames.size());
  std::vector<double> backupRho(m_points.size());
  for (int it = 0; it < maxIterations; ++it) {
    for (auto& f : m_frames)
      if (!f.inPrior) f.linearization = f.params;
    System sys = linearize(makePairs(), {});
    addPriors(sys);
    for (size_t i = 0; i < m_points.size(); ++i) m_points[i].hRho = sys.points[i].Hrr;

    Eigen::MatrixXd A = sys.H;
    A.diagonal() *= 1.0 + lambda;
    A.diagonal().array() += 1e-8;
    Eigen::VectorXd b = sys.g;
    std::vector<double> hr(m_points.size());
    std::vector<Eigen::MatrixXd> chunkA(kParallelChunks);
    std::vector<Eigen::VectorXd> chunkB(kParallelChunks);
    parallelChunks(m_points.size(), [&](size_t c, size_t begin, size_t end) {
      chunkA[c] = Eigen::MatrixXd::Zero(A.rows(), A.cols());
      chunkB[c] = Eigen::VectorXd::Zero(b.size());
      for (size_t i = begin; i < end; ++i) {
        hr[i] = sys.points[i].Hrr * (1.0 + lambda) + 1e-12;
        schurPoint(sys.points[i], hr[i], chunkA[c], chunkB[c]);
      }
    });
    for (size_t c = 0; c < kParallelChunks; ++c) {
      A += chunkA[c];
      b += chunkB[c];
    }
    const Eigen::VectorXd dx = -A.ldlt().solve(b);

    for (size_t i = 0; i < m_frames.size(); ++i) {
      backupFrames[i] = m_frames[i].params;
      KeyframeParams& fp = m_frames[i].params;
      fp.T_b_w = Sophus::SE3d::exp(dx.segment<6>(d * i)) * fp.T_b_w;
      for (int c = 0; c < m_rig.size(); ++c) {
        fp.affine[c].a += dx[d * i + 6 + 2 * c];
        fp.affine[c].b += dx[d * i + 7 + 2 * c];
      }
    }
    for (size_t i = 0; i < m_points.size(); ++i) {
      backupRho[i] = m_points[i].rho;
      const double step = (sys.points[i].gr + sys.points[i].Hfr.dot(dx)) / hr[i];
      const double rho = m_points[i].rho - step;
      m_points[i].rho = rho > 0 ? rho : 0.5 * m_points[i].rho;
    }

    // Given the frames, points are independent: a point keeps its depth step (or half of it) only if that lowers
    // its own energy. The photometric error is far from linear in depth; without this about half of the points
    // overshoot in every step and one bad point rejects the whole step, so the window never converges (KITTI 00).
    ++result.iterations;
    const Pairs pairs = makePairs();
    std::vector<double> chunkEnergy(kParallelChunks, 0.0);
    std::vector<int> chunkReverted(kParallelChunks, 0);
    parallelChunks(m_points.size(), [&](size_t c, size_t begin, size_t end) {
      for (size_t i = begin; i < end; ++i) {
        WindowPoint& p = m_points[i];
        double e = pointEnergy(pairs, p, p.rho);
        if (const double eOld = pointEnergy(pairs, p, backupRho[i]); eOld < e) {
          const double half = 0.5 * (p.rho + backupRho[i]);
          const double eHalf = pointEnergy(pairs, p, half);
          if (eHalf < eOld) {
            p.rho = half;
            e = eHalf;
          } else {
            p.rho = backupRho[i];
            e = eOld;
            ++chunkReverted[c];
          }
        }
        chunkEnergy[c] += e;
      }
    });
    double next = priorEnergy();
    int reverted = 0;
    for (size_t c = 0; c < kParallelChunks; ++c) {
      next += chunkEnergy[c];
      reverted += chunkReverted[c];
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
  const int d = frameDim();
  const Eigen::Index nf = static_cast<Eigen::Index>(m_frames.size());
  for (auto& f : m_frames)
    if (!f.inPrior) f.linearization = f.params;
  classifyResiduals();

  const Pairs pairs = makePairs();
  std::vector<size_t> hosted;
  for (size_t i = 0; i < m_points.size(); ++i)
    if (m_points[i].host == frameId) hosted.push_back(i);
  // Points with too little depth information are dropped rather than marginalised.
  std::vector<size_t> kept;
  if (!hosted.empty()) {
    const System all = linearize(pairs, hosted);
    for (size_t j = 0; j < hosted.size(); ++j)
      if (all.points[j].Hrr >= m_settings.minRhoHessian) kept.push_back(hosted[j]);
  }
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(d * nf, d * nf);
  Eigen::VectorXd g = Eigen::VectorXd::Zero(d * nf);
  if (!kept.empty()) {
    System sys = linearize(pairs, kept);
    for (const PointBlock& pb : sys.points) schurPoint(pb, pb.Hrr, sys.H, sys.g);
    H = std::move(sys.H);
    g = std::move(sys.g);
  }
  Eigen::VectorXd delta(d * nf);
  for (Eigen::Index i = 0; i < nf; ++i) delta.segment(d * i, d) = m_frames[i].delta();
  // The linearisation is around the current state; the prior is expressed in delta coordinates.
  m_priorH += H;
  m_priorB += g - H * delta;
  for (Eigen::Index i = 0; i < nf; ++i)
    if (!m_frames[i].inPrior && !m_priorH.block(d * i, d * i, d, d).isZero(0)) m_frames[i].inPrior = true;

  std::vector<Eigen::Index> keep;
  for (Eigen::Index i = 0; i < d * nf; ++i)
    if (i / d != k) keep.push_back(i);
  const Eigen::Index nk = static_cast<Eigen::Index>(keep.size());
  Eigen::MatrixXd Hrr(nk, nk), Hrk(nk, d);
  Eigen::VectorXd br(nk);
  for (Eigen::Index i = 0; i < nk; ++i) {
    br[i] = m_priorB[keep[i]];
    Hrk.row(i) = m_priorH.block(keep[i], d * k, 1, d);
    for (Eigen::Index j = 0; j < nk; ++j) Hrr(i, j) = m_priorH(keep[i], keep[j]);
  }
  Eigen::MatrixXd Hkk = m_priorH.block(d * k, d * k, d, d);
  Hkk.diagonal().array() += 1e-9 * std::max(Hkk.trace(), 1.0);
  const Eigen::MatrixXd HkkInv = Hkk.ldlt().solve(Eigen::MatrixXd::Identity(d, d));
  const Eigen::MatrixXd reduced = Hrr - Hrk * HkkInv * Hrk.transpose();
  const Eigen::VectorXd bk = m_priorB.segment(d * k, d);
  m_priorB = br - Hrk * (HkkInv * bk);
  m_priorH = 0.5 * (reduced + reduced.transpose());

  std::erase_if(m_points, [&](const WindowPoint& p) { return p.host == frameId; });
  for (auto& p : m_points)
    std::erase_if(p.residuals, [&](const WindowResidual& r) { return r.target == frameId; });
  m_frames.erase(m_frames.begin() + k);
}

void WindowOptimizer::logStaticDepthBias() const {
  if (m_rig.size() < 2) return;
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
  const Pairs pairs = makePairs();
  for (const auto& p : m_points) {
    if (p.rho <= 0) continue;
    std::vector<const WindowResidual*> statics;
    for (const auto& r : p.residuals)
      if (r.target == p.host && r.state == ResidualState::Good) statics.push_back(&r);
    if (statics.empty()) continue;
    const auto ls = argminLog(
        [&](double rho) {
          double total = 0, e;
          for (const WindowResidual* r : statics) {
            if (!residualEnergy(pairs, p, *r, rho, e)) return inf;
            total += e;
          }
          return total;
        },
        p.rho);
    if (!ls) continue;
    const double rhoStatic = p.rho * std::exp(*ls);
    const Camera& cam = m_rig.cameras[p.hostCam];
    const double radius =
        std::hypot(p.pattern.uv.x() - cam.cx, p.pattern.uv.y() - cam.cy) / std::hypot(cam.width / 2.0, cam.height / 2.0);
    const int bin = std::min(2, static_cast<int>(radius * 3));
    for (const auto& r : p.residuals) {
      if (r.target == p.host || r.state != ResidualState::Good) continue;
      const auto lt = argminLog(
          [&](double rho) {
            double e;
            return residualEnergy(pairs, p, r, rho, e) ? e : inf;
          },
          rhoStatic);
      if (!lt) continue;
      sum[bin] += *lt;
      ++count[bin];
    }
  }
  auto mean = [&](int b) { return count[b] ? 100.0 * sum[b] / count[b] : 0.0; };
  spdlog::info("temporal vs static inverse depth by image radius (inner/middle/outer third): {:+.2f} {:+.2f} {:+.2f} % "
               "({} {} {} residuals)",
               mean(0), mean(1), mean(2), count[0], count[1], count[2]);
}

}  // namespace sdv
