#include <sdv/semi_dense_mapper.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <execution>
#include <map>
#include <numeric>
#include <stdexcept>

#include <sdv/parallel.h>

namespace sdv {

namespace {

using Clock = std::chrono::steady_clock;
double since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

PointSelectorSettings selectorSettings(const SemiDenseSettings& s) {
  PointSelectorSettings p;
  p.targetPoints = s.pointsPerImage;
  return p;
}

std::array<std::uint8_t, 3> rgbAt(const cv::Mat& image, int u, int v) {
  if (image.depth() == CV_16U) {
    const cv::Vec3w bgr = image.at<cv::Vec3w>(v, u);
    return {static_cast<std::uint8_t>(bgr[2] >> 8), static_cast<std::uint8_t>(bgr[1] >> 8),
            static_cast<std::uint8_t>(bgr[0] >> 8)};
  }
  const cv::Vec3b bgr = image.at<cv::Vec3b>(v, u);
  return {bgr[2], bgr[1], bgr[0]};
}

}  // namespace

SemiDenseMapper::SemiDenseMapper(const Rig& rig, SemiDenseSettings settings)
    : m_rig(rig), m_settings(std::move(settings)), m_selector(selectorSettings(m_settings)) {
  m_settings.trace.rhoMaxInit = 1.0 / m_settings.minDepth;
}

void SemiDenseMapper::addFrame(int frameIndex, const std::vector<cv::Mat>& images, const Sophus::SE3d& T_w_b,
                               const std::vector<AffineBrightness>& affine, bool host) {
  const int nc = m_rig.size();
  if (static_cast<int>(images.size()) != nc || static_cast<int>(affine.size()) != nc)
    throw std::invalid_argument("one image and one brightness per rig camera required");
  std::vector<std::shared_ptr<const ImagePyramid>> pyr(nc);
  std::vector<Sophus::SE3d> T_c_w(nc);
  auto t0 = Clock::now();
  for (int c = 0; c < nc; ++c) {
    pyr[c] = std::make_shared<const ImagePyramid>(toFloatGray(images[c]), 1);
    T_c_w[c] = m_rig.T_c_b[c] * T_w_b.inverse();
  }
  m_stats.pyramid += since(t0);
  if (m_settings.verify) m_frames.push_back({frameIndex, pyr, T_c_w, affine});

  t0 = Clock::now();
  traceFrame(frameIndex, pyr, T_c_w, affine);
  m_stats.trace += since(t0);
  t0 = Clock::now();
  while (!m_hosts.empty() && frameIndex - m_hosts.front().frameIndex >= m_settings.traceFrames) {
    close(m_hosts.front());
    m_hosts.pop_front();
  }
  m_stats.close += since(t0);
  t0 = Clock::now();
  if (host) createHosts(frameIndex, images, pyr, T_c_w, affine);
  m_stats.create += since(t0);
  const int oldest = m_hosts.empty() ? frameIndex : m_hosts.front().frameIndex;
  while (!m_frames.empty() && m_frames.front().frameIndex < oldest) m_frames.pop_front();
}

void SemiDenseMapper::createHosts(int frameIndex, const std::vector<cv::Mat>& images,
                                  const std::vector<std::shared_ptr<const ImagePyramid>>& pyr,
                                  const std::vector<Sophus::SE3d>& T_c_w, const std::vector<AffineBrightness>& affine) {
  const int nc = m_rig.size();
  std::vector<Host> fresh(nc);
  std::vector<int> cams(nc);
  std::iota(cams.begin(), cams.end(), 0);
  std::for_each(std::execution::par, cams.begin(), cams.end(), [&](int c) {
    const ImageLevel& img = pyr[c]->level(0);
    Host& h = fresh[c];
    h = {frameIndex, c, T_c_w[c], affine[c], {}, {}, {}, images[c].channels() == 3};
    for (const auto& cand : m_selector.select(img, m_rig.cameras[c].maskImage())) {
      auto p = ImmaturePoint::create(m_rig.cameras[c], img, cand.uv.cast<double>(), m_settings.trace);
      if (!p) continue;
      h.points.push_back(std::move(*p));
      h.intensity.push_back(img.at(cand.uv.x(), cand.uv.y())[0]);
      if (h.hasColor) h.color.push_back(rgbAt(images[c], cand.uv.x(), cand.uv.y()));
    }
  });
  // The other cameras of the same frame: fixed extrinsic baseline, metric even where the rig barely moves. One pass
  // over the new points, each into the other cameras in order.
  std::vector<size_t> start{0};
  std::vector<HostTargetState> states;
  for (const Host& h : fresh) {
    m_stats.candidates += static_cast<long long>(h.points.size());
    start.push_back(start.back() + h.points.size());
    for (int c = 0; c < nc; ++c) {
      HostTargetState s;
      s.T_t_h = T_c_w[c] * h.T_c_w.inverse();
      s.host = h.affine;
      s.target = affine[c];
      states.push_back(s);
    }
  }
  std::array<long long, kParallelChunks> traces{}, good{};
  parallelChunks(start.back(), [&](size_t chunk, size_t begin, size_t end) {
    size_t hi = static_cast<size_t>(std::upper_bound(start.begin(), start.end(), begin) - start.begin()) - 1;
    for (size_t g = begin; g < end; ++g) {
      while (g >= start[hi + 1]) ++hi;
      ImmaturePoint& p = fresh[hi].points[g - start[hi]];
      for (int c = 0; c < nc; ++c)
        if (c != static_cast<int>(hi))
          tracePoint(p, m_rig.cameras[c], pyr[c]->level(0), states[hi * nc + c], false, traces[chunk], good[chunk]);
    }
  });
  for (size_t k = 0; k < kParallelChunks; ++k) m_stats.traces += traces[k], m_stats.good += good[k];
  for (Host& h : fresh) m_hosts.push_back(std::move(h));
}

// One trace of p into a target image; counts traces that saw the point and good ones.
void SemiDenseMapper::tracePoint(ImmaturePoint& p, const Camera& cam, const ImageLevel& img,
                                 const HostTargetState& state, bool requireVisible, long long& traces,
                                 long long& good) const {
  if (p.numOutliers() > p.numGood() + 2) return;
  if (requireVisible) {
    // Another camera at a later time sees only a small part of the host image; skip the others cheaply.
    Eigen::Vector2d uv;
    if (p.numGood() == 0 || !projectBearing(p.bearing(), p.rho(), state.T_t_h, cam, uv) ||
        !cam.isInside(uv.x(), uv.y(), 4.0))
      return;
  }
  // A target that cannot see the point must not undo what the others matched.
  ImmaturePoint traced = p;
  const TraceStatus s = traced.trace(cam, img, state, m_settings.trace);
  if (s == TraceStatus::OutOfBounds || s == TraceStatus::Skipped) return;
  ++traces;
  good += s == TraceStatus::Good;
  p = std::move(traced);
}

// All open hosts into all cameras of a new frame, in one parallel pass over their points (each point still visits
// the cameras in order, so the result is that of one pass per host and camera).
void SemiDenseMapper::traceFrame(int frameIndex, const std::vector<std::shared_ptr<const ImagePyramid>>& pyr,
                                 const std::vector<Sophus::SE3d>& T_c_w, const std::vector<AffineBrightness>& affine) {
  const int nc = m_rig.size();
  std::vector<size_t> start{0};
  std::vector<HostTargetState> states;
  for (const Host& h : m_hosts) {
    start.push_back(start.back() + h.points.size());
    for (int c = 0; c < nc; ++c) {
      HostTargetState s;
      s.T_t_h = T_c_w[c] * h.T_c_w.inverse();
      s.host = h.affine;
      s.target = affine[c];
      states.push_back(s);
    }
  }
  std::array<long long, kParallelChunks> traces{}, good{};
  parallelChunks(start.back(), [&](size_t chunk, size_t begin, size_t end) {
    size_t hi = static_cast<size_t>(std::upper_bound(start.begin(), start.end(), begin) - start.begin()) - 1;
    for (size_t g = begin; g < end; ++g) {
      while (g >= start[hi + 1]) ++hi;
      Host& h = m_hosts[hi];
      ImmaturePoint& p = h.points[g - start[hi]];
      if (m_settings.dropFrames > 0 && p.numGood() == 0 && frameIndex - h.frameIndex >= m_settings.dropFrames) continue;
      for (int c = 0; c < nc; ++c)
        tracePoint(p, m_rig.cameras[c], pyr[c]->level(0), states[hi * nc + c], c != h.camera, traces[chunk],
                   good[chunk]);
    }
  });
  for (size_t k = 0; k < kParallelChunks; ++k) m_stats.traces += traces[k], m_stats.good += good[k];
}

bool SemiDenseMapper::verify(const Host& h, const ImmaturePoint& p, double& rho) const {
  struct View {
    HostTargetState state;
    const Camera* cam;
    const ImageLevel* img;
  };
  std::vector<View> views;
  for (const BufferedFrame& f : m_frames)
    for (int c = 0; c < m_rig.size(); ++c) {
      if (f.frameIndex == h.frameIndex && c == h.camera) continue;
      HostTargetState s;
      s.T_t_h = f.T_c_w[c] * h.T_c_w.inverse();
      s.host = h.affine;
      s.target = f.affine[c];
      views.push_back({s, &m_rig.cameras[c], &f.pyr[c]->level(0)});
    }
  const PhotometricSettings& ps = m_settings.trace.photometric;
  const double maxEnergy = kPatternSize * 0.5 * m_settings.verifyMaxError * m_settings.verifyMaxError;
  PatternResidual res;
  auto rmse = [&](const PatternResidual& r) {
    double e = 0;
    for (const auto& px : r.pixels) e += px.r * px.r;
    return std::sqrt(e / kPatternSize);
  };

  // Gauss-Newton on the inverse depth over the views that match at the current estimate (the others are occluded
  // or wrong); the step is kept only if the energy of that set falls.
  const double rhoMin = 0.5 * p.rhoMin(), rhoMax = 2.0 * p.rhoMax();
  for (int it = 0; it < m_settings.verifyIterations; ++it) {
    std::vector<const View*> inliers;
    double H = 0, b = 0, energy = 0;
    for (const View& v : views) {
      if (!evaluatePatternResidual(p.pattern(), rho, v.state, *v.cam, *v.img, ps, res) || res.energy > 2 * maxEnergy)
        continue;
      inliers.push_back(&v);
      energy += res.energy;
      for (const auto& px : res.pixels) H += px.weight * px.dRho * px.dRho, b += px.weight * px.dRho * px.r;
    }
    if (inliers.empty() || H <= 1e-12) break;
    const double candidate = std::clamp(rho - b / H, rhoMin, rhoMax);
    double newEnergy = 0;
    bool ok = true;
    for (const View* v : inliers) {
      if (!evaluatePatternResidual(p.pattern(), candidate, v->state, *v->cam, *v->img, ps, res)) {
        ok = false;
        break;
      }
      newEnergy += res.energy;
    }
    if (!ok || newEnergy >= energy) break;
    rho = candidate;
  }

  int good = 0, informative = 0;
  for (const View& v : views) {
    if (!evaluatePatternResidual(p.pattern(), rho, v.state, *v.cam, *v.img, ps, res)) continue;
    Eigen::Vector2d uv, dRho;
    if (!projectBearing(p.bearing(), rho, v.state.T_t_h, *v.cam, uv, nullptr, &dRho)) continue;
    if (0.1 * rho * dRho.norm() < m_settings.verifyMinParallax) continue;
    ++informative;
    good += rmse(res) < m_settings.verifyMaxError;
  }
  return good >= m_settings.verifyMinViews && good >= m_settings.verifyMinFraction * informative;
}

void SemiDenseMapper::close(Host& h) {
  const Sophus::SE3d T_w_c = h.T_c_w.inverse();
  std::vector<double> rhos(h.points.size());
  std::vector<char> status(h.points.size(), 0);  // 0 accepted, 1 matches, 2 interval, 3 verification
  parallelChunks(h.points.size(), [&](size_t, size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i) {
      const ImmaturePoint& p = h.points[i];
      double rho = p.rho();
      if (p.numGood() < m_settings.minGood || p.numOutliers() > m_settings.maxOutlierRatio * p.numGood() ||
          rho <= 1e-6 || rho < p.rhoMin() || rho > p.rhoMax())
        status[i] = 1;
      else if (0.5 * (p.rhoMax() - p.rhoMin()) / rho > m_settings.maxInterval)
        status[i] = 2;
      else if (m_settings.verify && !verify(h, p, rho))
        status[i] = 3;
      rhos[i] = rho;
    }
  });
  for (size_t i = 0; i < h.points.size(); ++i) {
    const ImmaturePoint& p = h.points[i];
    if (status[i] == 1) ++m_stats.rejectMatches;
    if (status[i] == 2) ++m_stats.rejectInterval;
    if (status[i] == 3) ++m_stats.rejectVerify;
    if (status[i] != 0) continue;
    const double rho = rhos[i];
    const double interval = 0.5 * (p.rhoMax() - p.rhoMin()) / p.rho();
    MapPoint m{T_w_c * (p.bearing() / rho), h.intensity[i], h.frameIndex, h.camera, p.pattern().uv, 1.0 / rho,
               p.numGood(), interval, MapPointSource::SemiDense};
    if (h.hasColor) m.color = h.color[i];
    m_points.push_back(m);
  }
  m_stats.accepted = static_cast<long long>(m_points.size());
  h.points.clear();
}

std::vector<MapPoint> SemiDenseMapper::finish() {
  for (Host& h : m_hosts) close(h);
  m_hosts.clear();
  if (m_settings.voxelSize <= 0) return std::move(m_points);

  // The number of distinct host images in a voxel is a multi-view consistency check, since independently traced
  // hosts agree only on real surfaces.
  struct Cell {
    size_t best;
    std::vector<std::pair<int, int>> hosts;
    std::vector<size_t> members;
  };
  std::map<std::array<long long, 3>, Cell> grid;
  for (size_t i = 0; i < m_points.size(); ++i) {
    const Eigen::Vector3d& x = m_points[i].position;
    const std::array<long long, 3> key{static_cast<long long>(std::floor(x.x() / m_settings.voxelSize)),
                                       static_cast<long long>(std::floor(x.y() / m_settings.voxelSize)),
                                       static_cast<long long>(std::floor(x.z() / m_settings.voxelSize))};
    auto [it, inserted] = grid.try_emplace(key, Cell{i, {}, {}});
    Cell& cell = it->second;
    if (m_points[i].relativeDepthSigma < m_points[cell.best].relativeDepthSigma) cell.best = i;
    const std::pair<int, int> host{m_points[i].frameIndex, m_points[i].camera};
    if (std::find(cell.hosts.begin(), cell.hosts.end(), host) == cell.hosts.end()) cell.hosts.push_back(host);
    cell.members.push_back(i);
  }
  std::vector<MapPoint> out;
  for (const auto& [key, cell] : grid) {
    if (static_cast<int>(cell.hosts.size()) < m_settings.minVoxelHosts) continue;
    if (m_settings.thin) out.push_back(m_points[cell.best]);
    else
      for (size_t i : cell.members) out.push_back(m_points[i]);
  }
  m_stats.merged = static_cast<long long>(out.size());
  m_points.clear();
  return out;
}

}  // namespace sdv
