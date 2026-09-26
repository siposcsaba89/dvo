#include <sdv/odometry.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace sdv {

namespace {

PointSelectorSettings selectorSettings(const OdometrySettings& s) {
  PointSelectorSettings p;
  p.targetPoints = s.candidatesPerKeyframe;
  return p;
}

}  // namespace

MonoOdometry::MonoOdometry(const Camera& cam, OdometrySettings settings)
    : m_camera(cam),
      m_settings(std::move(settings)),
      m_initializer(cam, m_settings.levels, m_settings.init),
      m_tracker(m_settings.tracking),
      m_window(cam, m_settings.window),
      m_selector(selectorSettings(m_settings)),
      m_activationCell(m_settings.initialActivationCell) {}

OdometryFrameInfo MonoOdometry::addFrame(const cv::Mat& image) {
  OdometryFrameInfo info;
  const int index = m_frameCount++;
  m_frames.emplace_back();
  auto pyr = std::make_shared<const ImagePyramid>(toFloatGray(image), m_settings.levels);

  if (!m_initialized) {
    if (!m_initHost) {
      m_initializer.reset(*pyr);
      m_initHost = pyr;
      m_initHostIndex = index;
      return info;
    }
    const MonoInitResult res = m_initializer.addFrame(*pyr);
    if (res.reset) {
      m_initHost = pyr;
      m_initHostIndex = index;
      return info;
    }
    if (!res.initialized) return info;
    initializeFromMono(res, pyr);
    info.initialized = info.keyframe = info.trackingOk = true;
    info.activePoints = static_cast<int>(m_window.points().size());
    return info;
  }

  const TrackingResult res = m_tracker.track(*m_reference, *pyr, makeMotionHypotheses(m_T_prev_ref, m_T_prev_prevprev),
                                             m_lastAffine);
  m_T_prev_prevprev = res.T_t_h * m_T_prev_ref.inverse();
  m_T_prev_ref = res.T_t_h;
  m_lastAffine = res.affine;
  m_frames.back() = {m_referenceId, res.T_t_h};
  if (m_referenceRmse < 0) m_referenceRmse = res.rmse;

  const Sophus::SE3d T_c_w = res.T_t_h * m_window.frame(m_referenceId).params.T_c_w;
  traceCandidates(*pyr, T_c_w, res.affine);
  info.initialized = true;
  info.trackingOk = res.ok;
  info.rmse = res.rmse;
  if (needKeyframe(res)) {
    const int id = createKeyframe(pyr, T_c_w, res.affine);
    m_frames.back() = {id, Sophus::SE3d()};
    info.keyframe = true;
  }
  info.activePoints = static_cast<int>(m_window.points().size());
  for (const auto& [id, kf] : m_keyframes) info.immaturePoints += static_cast<int>(kf.immature.size());
  return info;
}

void MonoOdometry::initializeFromMono(const MonoInitResult& res, std::shared_ptr<const ImagePyramid> current) {
  const int host = m_window.addFrame(m_initHost, Sophus::SE3d());
  m_keyframes[host] = {m_initHostIndex, m_initHost, {}};
  m_keyframeFrameIndex[host] = m_initHostIndex;
  m_frames[m_initHostIndex] = {host, Sophus::SE3d()};
  for (const auto& p : m_initializer.points())
    if (p.relativeSigma <= m_settings.initMaxSigma) m_window.addPoint(host, p.uv, p.rho);
  selectCandidates(host);
  m_initialized = true;

  traceCandidates(*current, res.T_t_h, res.affine);
  const int id = createKeyframe(current, res.T_t_h, res.affine);
  m_frames.back() = {id, Sophus::SE3d()};
}

int MonoOdometry::createKeyframe(std::shared_ptr<const ImagePyramid> image, const Sophus::SE3d& T_c_w,
                                 const AffineBrightness& affine) {
  const int index = m_frameCount - 1;
  const int id = m_window.addFrame(image, T_c_w, affine);
  m_keyframes[id] = {index, image, {}};
  m_keyframeFrameIndex[id] = index;

  activateCandidates(id);
  m_window.optimize(m_settings.windowIterations);
  removeOutlierPoints();
  storeKeyframePoses();
  marginalizeKeyframes();
  selectCandidates(id);

  m_referenceId = id;
  m_lastAffine = m_window.frame(id).params.affine;
  buildReference();
  m_T_prev_ref = Sophus::SE3d();
  m_referenceRmse = -1;
  return id;
}

void MonoOdometry::selectCandidates(int keyframeId) {
  Keyframe& kf = m_keyframes.at(keyframeId);
  const ImageLevel& img = kf.image->level(0);
  for (const auto& c : m_selector.select(img))
    if (auto p = ImmaturePoint::create(m_camera, img, c.uv.cast<double>(), m_settings.trace))
      kf.immature.push_back(std::move(*p));
}

void MonoOdometry::traceCandidates(const ImagePyramid& image, const Sophus::SE3d& T_c_w,
                                   const AffineBrightness& affine) {
  for (auto& [id, kf] : m_keyframes) {
    const WindowFrame& host = m_window.frame(id);
    HostTargetState state;
    state.T_t_h = T_c_w * host.params.T_c_w.inverse();
    state.host = host.params.affine;
    state.target = affine;
    for (auto& p : kf.immature) p.trace(m_camera, image.level(0), state, m_settings.trace);
    std::erase_if(kf.immature, [](const ImmaturePoint& p) {
      return p.lastStatus() == TraceStatus::OutOfBounds || p.numOutliers() > p.numGood() + 2;
    });
  }
}

void MonoOdometry::activateCandidates(int newKeyframeId) {
  const WindowFrame& target = m_window.frame(newKeyframeId);
  const int cell = std::max(2, static_cast<int>(std::lround(m_activationCell)));
  const int gw = (m_camera.width + cell - 1) / cell, gh = (m_camera.height + cell - 1) / cell;
  std::vector<char> occupied(static_cast<size_t>(gw) * gh, 0);
  auto cellOf = [&](const Sophus::SE3d& T_t_h, const Eigen::Vector3d& bearing, double rho) {
    Eigen::Vector2d uv;
    if (!projectBearing(bearing, rho, T_t_h, m_camera, uv) || !m_camera.isInside(uv.x(), uv.y(), 0.0)) return -1;
    return static_cast<int>(uv.y()) / cell * gw + static_cast<int>(uv.x()) / cell;
  };

  for (const auto& p : m_window.points()) {
    const Sophus::SE3d T_t_h = target.params.T_c_w * m_window.frame(p.host).params.T_c_w.inverse();
    if (const int c = cellOf(T_t_h, p.bearing, p.rho); c >= 0) occupied[c] = 1;
  }

  for (auto& [id, kf] : m_keyframes) {
    if (id == newKeyframeId) continue;
    const Sophus::SE3d T_t_h = target.params.T_c_w * m_window.frame(id).params.T_c_w.inverse();
    std::erase_if(kf.immature, [&](const ImmaturePoint& p) {
      if (p.lastStatus() != TraceStatus::Good || p.numGood() < m_settings.activationMinGood ||
          p.lastErrorPixels() > m_settings.activationMaxErrorPixels || p.rho() <= 0)
        return false;
      const int c = cellOf(T_t_h, p.bearing(), p.rho());
      if (c < 0 || occupied[c]) return false;
      occupied[c] = 1;
      return m_window.addPoint(id, p.pattern().uv, p.rho()) >= 0;
    });
  }

  const double ratio = static_cast<double>(m_window.points().size()) / m_settings.targetActivePoints;
  m_activationCell = std::clamp(m_activationCell * std::sqrt(std::max(ratio, 0.25)), 3.0, 40.0);
}

void MonoOdometry::removeOutlierPoints() {
  std::vector<int> remove;
  for (const auto& p : m_window.points()) {
    const int good = p.numGood();
    const auto outliers = std::count_if(p.residuals.begin(), p.residuals.end(),
                                        [](const WindowResidual& r) { return r.state == ResidualState::Outlier; });
    if ((good == 0 && !p.residuals.empty()) || outliers > good) remove.push_back(p.id);
  }
  for (int id : remove) m_window.removePoint(id);
}

void MonoOdometry::marginalizeKeyframes() {
  {
    const auto& frames = m_window.frames();
    if (frames.size() <= 2) return;
    const int newest = frames.back().id;
    std::vector<int> flagged;
    for (size_t i = 0; i + 2 < frames.size(); ++i) {
      int hosted = 0, visible = 0;
      for (const auto& p : m_window.points()) {
        if (p.host != frames[i].id) continue;
        ++hosted;
        for (const auto& r : p.residuals) visible += r.target == newest && r.state == ResidualState::Good;
      }
      if (visible < m_settings.marginalizeVisibleFraction * std::max(hosted, 1)) flagged.push_back(frames[i].id);
    }
    for (int id : flagged) marginalize(id);
  }

  while (static_cast<int>(m_window.frames().size()) > m_settings.maxKeyframes) {
    const auto& frames = m_window.frames();
    const size_t n = frames.size();
    auto center = [&](size_t i) { return frames[i].params.T_c_w.inverse().translation(); };
    // Distance score, DSO §3.1: keeps keyframes close to the newest one while spreading out the others.
    constexpr double eps = 1e-4;
    size_t best = 0;
    double bestScore = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i + 2 < n; ++i) {
      double sum = 0;
      for (size_t j = 0; j + 2 < n; ++j)
        if (j != i) sum += 1.0 / ((center(i) - center(j)).norm() + eps);
      const double score = std::sqrt((center(i) - center(n - 1)).norm()) * sum;
      if (score > bestScore) {
        bestScore = score;
        best = i;
      }
    }
    marginalize(frames[best].id);
  }
}

void MonoOdometry::marginalize(int keyframeId) {
  const WindowFrame& f = m_window.frame(keyframeId);
  const Sophus::SE3d T_w_c = f.params.T_c_w.inverse();
  const ImageLevel& img = f.image->level(0);
  for (const auto& p : m_window.points())
    if (p.host == keyframeId && p.numGood() > 0)
      m_marginalizedPoints.push_back({T_w_c * (p.bearing / p.rho),
                                      img.interpolateIntensity(static_cast<float>(p.pattern.uv.x()),
                                                               static_cast<float>(p.pattern.uv.y())),
                                      m_keyframeFrameIndex.at(keyframeId), p.pattern.uv});
  m_keyframePoses[keyframeId] = f.params.T_c_w;
  m_window.marginalizeFrame(keyframeId);
  m_keyframes.erase(keyframeId);
}

void MonoOdometry::buildReference() {
  const WindowFrame& kf = m_window.frame(m_referenceId);
  std::vector<Eigen::Vector2i> pixels;
  std::vector<double> rhos;
  for (const auto& p : m_window.points()) {
    if (p.host != kf.id && p.numGood() == 0) continue;
    const Sophus::SE3d T_kf_h = kf.params.T_c_w * m_window.frame(p.host).params.T_c_w.inverse();
    const Eigen::Vector3d x = T_kf_h.so3() * p.bearing + p.rho * T_kf_h.translation();
    Eigen::Vector2d uv;
    if (!m_camera.project(x, uv) || !m_camera.isInside(uv.x(), uv.y(), 2.0)) continue;
    pixels.emplace_back(static_cast<int>(std::lround(uv.x())), static_cast<int>(std::lround(uv.y())));
    rhos.push_back(p.rho / x.norm());
  }
  m_reference.emplace(m_camera, *kf.image, pixels, rhos, kf.params.affine, m_settings.tracking.gradientWeightC);
}

double MonoOdometry::translationFlow(const Sophus::SE3d& T_f_ref) const {
  const auto& pts = m_reference->points(0);
  double sum = 0;
  int n = 0;
  Eigen::Vector2d full, rot;
  for (size_t i = 0; i < pts.size(); i += 4) {
    const Eigen::Vector3d r = T_f_ref.so3() * pts[i].bearing;
    if (!m_camera.project(Eigen::Vector3d(r + pts[i].rho * T_f_ref.translation()), full) ||
        !m_camera.project(r, rot))
      continue;
    sum += (full - rot).norm();
    ++n;
  }
  return n > 0 ? sum / n : 0.0;
}

bool MonoOdometry::needKeyframe(const TrackingResult& res) const {
  const double brightness = std::abs(res.affine.a - m_window.frame(m_referenceId).params.affine.a);
  const double score = res.meanFlow / m_settings.kfFlow + translationFlow(res.T_t_h) / m_settings.kfTranslationFlow +
                       brightness / m_settings.kfBrightness;
  return score > 1.0 || (m_referenceRmse > 0 && res.rmse > m_settings.kfRmseFactor * m_referenceRmse);
}

void MonoOdometry::storeKeyframePoses() {
  for (const auto& f : m_window.frames()) m_keyframePoses[f.id] = f.params.T_c_w;
}

std::vector<std::optional<Sophus::SE3d>> MonoOdometry::poses() const {
  std::map<int, Sophus::SE3d> current = m_keyframePoses;
  for (const auto& f : m_window.frames()) current[f.id] = f.params.T_c_w;
  std::vector<std::optional<Sophus::SE3d>> out;
  for (const auto& r : m_frames)
    out.push_back(r.keyframe < 0 ? std::nullopt : std::optional((r.T_f_kf * current.at(r.keyframe)).inverse()));
  return out;
}

std::vector<MapPoint> MonoOdometry::mapPoints() const {
  std::vector<MapPoint> out = m_marginalizedPoints;
  for (const auto& p : m_window.points()) {
    if (p.numGood() == 0) continue;
    const WindowFrame& f = m_window.frame(p.host);
    const ImageLevel& img = f.image->level(0);
    out.push_back({f.params.T_c_w.inverse() * (p.bearing / p.rho),
                   img.interpolateIntensity(static_cast<float>(p.pattern.uv.x()), static_cast<float>(p.pattern.uv.y())),
                   m_keyframeFrameIndex.at(p.host), p.pattern.uv});
  }
  return out;
}

std::vector<int> MonoOdometry::keyframeIndices() const {
  std::vector<int> out;
  for (const auto& [id, index] : m_keyframeFrameIndex) out.push_back(index);
  return out;
}

}  // namespace sdv
