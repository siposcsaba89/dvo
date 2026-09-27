#include <sdv/semi_dense_mapper.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

#include <sdv/parallel.h>

namespace sdv {

namespace {

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
  for (int c = 0; c < nc; ++c) {
    pyr[c] = std::make_shared<const ImagePyramid>(toFloatGray(images[c]), 1);
    T_c_w[c] = m_rig.T_c_b[c] * T_w_b.inverse();
  }

  for (Host& h : m_hosts)
    for (int c = 0; c < nc; ++c) traceInto(h, c, pyr[c]->level(0), T_c_w[c], affine[c], c != h.camera);
  while (!m_hosts.empty() && frameIndex - m_hosts.front().frameIndex >= m_settings.traceFrames) {
    close(m_hosts.front());
    m_hosts.pop_front();
  }
  if (host) createHosts(frameIndex, images, pyr, T_c_w, affine);
}

void SemiDenseMapper::createHosts(int frameIndex, const std::vector<cv::Mat>& images,
                                  const std::vector<std::shared_ptr<const ImagePyramid>>& pyr,
                                  const std::vector<Sophus::SE3d>& T_c_w, const std::vector<AffineBrightness>& affine) {
  const int nc = m_rig.size();
  for (int c = 0; c < nc; ++c) {
    const ImageLevel& img = pyr[c]->level(0);
    Host h{frameIndex, c, T_c_w[c], affine[c], {}, {}, {}, images[c].channels() == 3};
    for (const auto& cand : m_selector.select(img, m_rig.cameras[c].maskImage())) {
      auto p = ImmaturePoint::create(m_rig.cameras[c], img, cand.uv.cast<double>(), m_settings.trace);
      if (!p) continue;
      h.points.push_back(std::move(*p));
      h.intensity.push_back(img.at(cand.uv.x(), cand.uv.y())[0]);
      if (h.hasColor) h.color.push_back(rgbAt(images[c], cand.uv.x(), cand.uv.y()));
    }
    m_stats.candidates += static_cast<long long>(h.points.size());
    // The other cameras of the same frame: fixed extrinsic baseline, metric even where the rig barely moves.
    for (int other = 0; other < nc; ++other)
      if (other != c) traceInto(h, other, pyr[other]->level(0), T_c_w[other], affine[other], false);
    m_hosts.push_back(std::move(h));
  }
}

void SemiDenseMapper::traceInto(Host& h, int camera, const ImageLevel& img, const Sophus::SE3d& T_c_w,
                                const AffineBrightness& affine, bool requireVisible) {
  HostTargetState state;
  state.T_t_h = T_c_w * h.T_c_w.inverse();
  state.host = h.affine;
  state.target = affine;
  const Camera& cam = m_rig.cameras[camera];
  std::array<long long, kParallelChunks> traces{}, good{};
  parallelChunks(h.points.size(), [&](size_t chunk, size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i) {
      ImmaturePoint& p = h.points[i];
      if (p.numOutliers() > p.numGood() + 2) continue;
      if (requireVisible) {
        // Another camera at a later time sees only a small part of the host image; skip the others cheaply.
        Eigen::Vector2d uv;
        if (p.numGood() == 0 || !projectBearing(p.bearing(), p.rho(), state.T_t_h, cam, uv) ||
            !cam.isInside(uv.x(), uv.y(), 4.0))
          continue;
      }
      // A target that cannot see the point must not undo what the others matched.
      ImmaturePoint traced = p;
      const TraceStatus s = traced.trace(cam, img, state, m_settings.trace);
      if (s == TraceStatus::OutOfBounds || s == TraceStatus::Skipped) continue;
      ++traces[chunk];
      good[chunk] += s == TraceStatus::Good;
      p = std::move(traced);
    }
  });
  for (size_t k = 0; k < kParallelChunks; ++k) m_stats.traces += traces[k], m_stats.good += good[k];
}

void SemiDenseMapper::close(Host& h) {
  const Sophus::SE3d T_w_c = h.T_c_w.inverse();
  for (size_t i = 0; i < h.points.size(); ++i) {
    const ImmaturePoint& p = h.points[i];
    const double rho = p.rho();
    if (p.numGood() < m_settings.minGood || p.numOutliers() > m_settings.maxOutlierRatio * p.numGood() ||
        rho <= 1e-6 || rho < p.rhoMin() || rho > p.rhoMax()) {
      ++m_stats.rejectMatches;
      continue;
    }
    const double interval = 0.5 * (p.rhoMax() - p.rhoMin()) / rho;
    if (interval > m_settings.maxInterval) {
      ++m_stats.rejectInterval;
      continue;
    }
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
