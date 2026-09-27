#include <sdv/odometry.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <spdlog/spdlog.h>

#include <sdv/parallel.h>

namespace sdv {

namespace {

// Points this close to infinity have no usable position for the map.
constexpr double kMinMapRho = 1e-6;

PointSelectorSettings selectorSettings(const OdometrySettings& s) {
  PointSelectorSettings p;
  p.targetPoints = s.candidatesPerKeyframe;
  return p;
}

}  // namespace

Odometry::Odometry(const Camera& cam, OdometrySettings settings)
    : m_camera(cam),
      m_settings(std::move(settings)),
      m_initializer(cam, m_settings.levels, m_settings.init),
      m_tracker(m_settings.tracking),
      m_window(cam, m_settings.window),
      m_selector(selectorSettings(m_settings)),
      m_activationCell(m_settings.initialActivationCell) {}

void Odometry::enableStereo(const Camera& rightCam, const Sophus::SE3d& T_r_l) {
  if (m_frameCount > 0) throw std::logic_error("enableStereo must be called before the first frame");
  m_rightCam = rightCam;
  m_T_r_l = T_r_l;
  m_window.setStereo(rightCam, T_r_l);
}

OdometryFrameInfo Odometry::addFrame(const cv::Mat& image, const cv::Mat& right) {
  if (m_rightCam && right.empty()) throw std::invalid_argument("stereo odometry needs the right image");
  OdometryFrameInfo info;
  const int index = m_frameCount++;
  m_frames.emplace_back();
  m_currentRight = right;
  std::shared_ptr<const ImagePyramid> pyr;
  {
    auto t = m_profile.scope("pyramid");
    pyr = std::make_shared<const ImagePyramid>(toFloatGray(image), m_settings.levels);
  }

  if (!m_initialized && m_rightCam) {
    m_initialized = true;
    const int id = createKeyframe(pyr, Sophus::SE3d(), {});
    m_frames.back() = {id, Sophus::SE3d()};
    info.initialized = info.keyframe = info.trackingOk = true;
    info.activePoints = static_cast<int>(m_window.points().size());
    return info;
  }

  if (!m_initialized) {
    auto t = m_profile.scope("mono init");
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

  TrackingResult res;
  {
    auto t = m_profile.scope("track");
    res = m_tracker.track(*m_reference, *pyr, makeMotionHypotheses(m_T_prev_ref, m_T_prev_prevprev), m_lastAffine);
  }
  m_T_prev_prevprev = res.T_t_h * m_T_prev_ref.inverse();
  m_T_prev_ref = res.T_t_h;
  m_lastAffine = res.affine;
  m_frames.back() = {m_referenceId, res.T_t_h};
  if (m_referenceRmse < 0) m_referenceRmse = res.rmse;

  const Sophus::SE3d T_c_w = res.T_t_h * m_window.frame(m_referenceId).params.T_c_w;
  {
    auto t = m_profile.scope("trace");
    traceCandidates(*pyr, T_c_w, res.affine);
  }
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

void Odometry::initializeFromMono(const MonoInitResult& res, std::shared_ptr<const ImagePyramid> current) {
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

int Odometry::createKeyframe(std::shared_ptr<const ImagePyramid> image, const Sophus::SE3d& T_c_w,
                                 const AffineBrightness& affine) {
  const int index = m_frameCount - 1;
  const int id = m_window.addFrame(image, T_c_w, affine);
  m_keyframes[id] = {index, image, {}};
  m_keyframeFrameIndex[id] = index;
  if (m_rightCam) {
    // Stereo: new candidates get a metric depth interval from the right image before activation.
    auto t = m_profile.scope("kf select+stereo");
    selectCandidates(id);
    traceStereo(id, std::make_shared<const ImagePyramid>(toFloatGray(m_currentRight), 1));
  }

  const size_t before = m_window.points().size();
  {
    auto t = m_profile.scope("kf activate");
    activateCandidates(id);
  }
  const size_t activated = m_window.points().size();
  {
    auto t = m_profile.scope("kf window BA");
    m_window.optimize(m_settings.windowIterations);
  }
  if (m_settings.checkCalibration) m_window.logStereoDepthBias();
  size_t kept;
  {
    auto t = m_profile.scope("kf marginalize");
    removeOutlierPoints();
    kept = m_window.points().size();
    storeKeyframePoses();
    marginalizeKeyframes();
  }
  spdlog::debug("keyframe {}: points {} +{} activated -{} outliers -{} marginalised, activation cell {:.1f}", index,
                before, activated - before, activated - kept, kept - m_window.points().size(), m_activationCell);
  if (!m_rightCam) {
    auto t = m_profile.scope("kf select");
    selectCandidates(id);
  }

  m_referenceId = id;
  m_lastAffine = m_window.frame(id).params.affine;
  auto t = m_profile.scope("kf reference");
  buildReference();
  m_T_prev_ref = Sophus::SE3d();
  m_referenceRmse = -1;
  return id;
}

TraceSettings Odometry::candidateSettings() const {
  TraceSettings s = m_settings.trace;
  if (m_rightCam) {
    s.rhoMaxInit = 1.0 / m_settings.stereoMinDepth;
    s.maxSamples = m_settings.stereoMaxSamples;
  }
  return s;
}

void Odometry::selectCandidates(int keyframeId) {
  Keyframe& kf = m_keyframes.at(keyframeId);
  const ImageLevel& img = kf.image->level(0);
  const TraceSettings settings = candidateSettings();
  for (const auto& c : m_selector.select(img, m_camera.maskImage()))
    if (auto p = ImmaturePoint::create(m_camera, img, c.uv.cast<double>(), settings))
      kf.immature.push_back(std::move(*p));
}

void Odometry::traceStereo(int keyframeId, std::shared_ptr<const ImagePyramid> right) {
  Keyframe& kf = m_keyframes.at(keyframeId);
  const ImageLevel& img = right->level(0);
  HostTargetState state;
  state.T_t_h = m_T_r_l;
  const TraceSettings settings = candidateSettings();
  parallelChunks(kf.immature.size(), [&](size_t, size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i) kf.immature[i].trace(*m_rightCam, img, state, settings);
  });

  // Left/right gain and offset from the matched patterns (least squares on I_r = g * I_l + b).
  double sx = 0, sy = 0, sxx = 0, sxy = 0;
  int n = 0;
  for (const auto& p : kf.immature) {
    if (p.lastStatus() != TraceStatus::Good) continue;
    for (int k = 0; k < kPatternSize; ++k) {
      Eigen::Vector2d uv;
      if (!projectBearing(p.pattern().bearings[k], p.rho(), m_T_r_l, *m_rightCam, uv) ||
          !m_rightCam->isInside(uv.x(), uv.y(), 1.0))
        continue;
      const double x = p.pattern().intensities[k];
      const double y = img.interpolateIntensity(static_cast<float>(uv.x()), static_cast<float>(uv.y()));
      sx += x;
      sy += y;
      sxx += x * x;
      sxy += x * y;
      ++n;
    }
  }
  AffineBrightness stereoAffine;
  const double det = n * sxx - sx * sx;
  if (n > 200 && det > 1e-9) {
    const double gain = (n * sxy - sx * sy) / det;
    if (gain > 0.5 && gain < 2.0) stereoAffine = {std::log(gain), (sy - gain * sx) / n};
  }
  m_window.setFrameStereo(keyframeId, std::move(right), stereoAffine);
}

void Odometry::traceCandidates(const ImagePyramid& image, const Sophus::SE3d& T_c_w,
                                   const AffineBrightness& affine) {
  for (auto& [id, kf] : m_keyframes) {
    const WindowFrame& host = m_window.frame(id);
    HostTargetState state;
    state.T_t_h = T_c_w * host.params.T_c_w.inverse();
    state.host = host.params.affine;
    state.target = affine;
    parallelChunks(kf.immature.size(), [&](size_t, size_t begin, size_t end) {
      for (size_t i = begin; i < end; ++i) kf.immature[i].trace(m_camera, image.level(0), state, m_settings.trace);
    });
    std::erase_if(kf.immature, [](const ImmaturePoint& p) {
      return p.lastStatus() == TraceStatus::OutOfBounds || p.numOutliers() > p.numGood() + 2;
    });
  }
}

void Odometry::activateCandidates(int newKeyframeId) {
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
    if (id == newKeyframeId && !m_rightCam) continue;  // in stereo mode its own candidates are stereo-matched
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

void Odometry::removeOutlierPoints() {
  // Outlier residuals are already excluded from the optimisation. A point is only dropped when no residual
  // supports it: new points are often occluded or strongly warped in keyframes older than their host, which
  // tracing never checked, and dropping them for that starves monocular scale (KITTI 00, frames 4000-4500).
  std::vector<int> remove;
  for (const auto& p : m_window.points())
    if (p.numGood() == 0 && !p.residuals.empty()) remove.push_back(p.id);
  for (int id : remove) m_window.removePoint(id);
}

void Odometry::marginalizeKeyframes() {
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

void Odometry::marginalize(int keyframeId) {
  const WindowFrame& f = m_window.frame(keyframeId);
  const Sophus::SE3d T_w_c = f.params.T_c_w.inverse();
  const ImageLevel& img = f.image->level(0);
  for (const auto& p : m_window.points())
    if (p.host == keyframeId && p.numGood() > 0 && p.rho > kMinMapRho)
      m_marginalizedPoints.push_back({T_w_c * (p.bearing / p.rho),
                                      img.interpolateIntensity(static_cast<float>(p.pattern.uv.x()),
                                                               static_cast<float>(p.pattern.uv.y())),
                                      m_keyframeFrameIndex.at(keyframeId), p.pattern.uv, 1.0 / p.rho});
  m_keyframePoses[keyframeId] = f.params.T_c_w;
  m_window.marginalizeFrame(keyframeId);
  m_keyframes.erase(keyframeId);
}

void Odometry::buildReference() {
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

double Odometry::translationFlow(const Sophus::SE3d& T_f_ref) const {
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

bool Odometry::needKeyframe(const TrackingResult& res) const {
  const double brightness = std::abs(res.affine.a - m_window.frame(m_referenceId).params.affine.a);
  const double score = res.meanFlow / m_settings.kfFlow + translationFlow(res.T_t_h) / m_settings.kfTranslationFlow +
                       brightness / m_settings.kfBrightness;
  return score > 1.0 || (m_referenceRmse > 0 && res.rmse > m_settings.kfRmseFactor * m_referenceRmse);
}

void Odometry::storeKeyframePoses() {
  for (const auto& f : m_window.frames()) m_keyframePoses[f.id] = f.params.T_c_w;
}

std::vector<std::optional<Sophus::SE3d>> Odometry::poses() const {
  std::map<int, Sophus::SE3d> current = m_keyframePoses;
  for (const auto& f : m_window.frames()) current[f.id] = f.params.T_c_w;
  std::vector<std::optional<Sophus::SE3d>> out;
  for (const auto& r : m_frames)
    out.push_back(r.keyframe < 0 ? std::nullopt : std::optional((r.T_f_kf * current.at(r.keyframe)).inverse()));
  return out;
}

std::vector<MapPoint> Odometry::mapPoints() const {
  std::vector<MapPoint> out = m_marginalizedPoints;
  for (const auto& p : m_window.points()) {
    if (p.numGood() == 0 || p.rho <= kMinMapRho) continue;
    const WindowFrame& f = m_window.frame(p.host);
    const ImageLevel& img = f.image->level(0);
    out.push_back({f.params.T_c_w.inverse() * (p.bearing / p.rho),
                   img.interpolateIntensity(static_cast<float>(p.pattern.uv.x()), static_cast<float>(p.pattern.uv.y())),
                   m_keyframeFrameIndex.at(p.host), p.pattern.uv, 1.0 / p.rho});
  }
  return out;
}

std::vector<int> Odometry::keyframeIndices() const {
  std::vector<int> out;
  for (const auto& [id, index] : m_keyframeFrameIndex) out.push_back(index);
  return out;
}

}  // namespace sdv
