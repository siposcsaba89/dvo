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

Odometry::Odometry(const Rig& rig, OdometrySettings settings)
    : m_rig(rig),
      m_settings(std::move(settings)),
      m_initializer(rig.cameras.at(0), m_settings.levels, m_settings.init),
      m_tracker(m_settings.tracking),
      m_window(rig, m_settings.window),
      m_selector(selectorSettings(m_settings)),
      m_lastAffine(rig.size()),
      m_activationCell(m_settings.initialActivationCell) {}

OdometryFrameInfo Odometry::addFrame(const std::vector<cv::Mat>& images) {
  if (static_cast<int>(images.size()) != m_rig.size()) throw std::invalid_argument("one image per rig camera required");
  OdometryFrameInfo info;
  const int index = m_frameCount++;
  m_frames.emplace_back();
  Pyramids pyr(m_rig.size());
  {
    auto t = m_profile.scope("pyramid");
    for (int c = 0; c < m_rig.size(); ++c)
      pyr[c] = std::make_shared<const ImagePyramid>(toFloatGray(images[c]), m_settings.levels);
  }

  if (!m_initialized && multiCamera()) {
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
      m_initializer.reset(*pyr[0]);
      m_initHost = pyr[0];
      m_initHostIndex = index;
      return info;
    }
    const MonoInitResult res = m_initializer.addFrame(*pyr[0]);
    if (res.reset) {
      m_initHost = pyr[0];
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
    std::vector<const ImagePyramid*> targets;
    for (const auto& p : pyr) targets.push_back(p.get());
    res = m_tracker.track(m_rig, m_reference, targets, makeMotionHypotheses(m_T_prev_ref, m_T_prev_prevprev),
                          m_lastAffine, &m_T_prev_ref);
  }
  m_T_prev_prevprev = res.T_t_h * m_T_prev_ref.inverse();
  m_T_prev_ref = res.T_t_h;
  m_lastAffine = res.affine;
  m_frames.back() = {m_referenceId, res.T_t_h, res.affine};
  if (m_referenceRmse < 0) m_referenceRmse = res.rmse;

  const Sophus::SE3d T_b_w = res.T_t_h * m_window.frame(m_referenceId).params.T_b_w;
  {
    auto t = m_profile.scope("trace");
    traceCandidates(pyr, T_b_w, res.affine);
  }
  info.initialized = true;
  info.trackingOk = res.ok;
  info.rmse = res.rmse;
  if (needKeyframe(res)) {
    const int id = createKeyframe(pyr, T_b_w, res.affine);
    m_frames.back() = {id, Sophus::SE3d()};
    info.keyframe = true;
  }
  info.activePoints = static_cast<int>(m_window.points().size());
  for (const auto& [id, kf] : m_keyframes)
    for (const auto& cam : kf.immature) info.immaturePoints += static_cast<int>(cam.size());
  return info;
}

void Odometry::initializeFromMono(const MonoInitResult& res, const Pyramids& current) {
  const int host = m_window.addFrame(m_initHost, Sophus::SE3d());
  m_keyframes[host] = {m_initHostIndex, {m_initHost}, std::vector<std::vector<ImmaturePoint>>(1)};
  m_keyframeFrameIndex[host] = m_initHostIndex;
  m_frames[m_initHostIndex] = {host, Sophus::SE3d()};
  for (const auto& p : m_initializer.points())
    if (p.relativeSigma <= m_settings.initMaxSigma) m_window.addPoint(host, p.uv, p.rho);
  selectCandidates(host);
  m_initialized = true;

  // The initialiser estimates the camera motion; the world frame is the body frame of the first keyframe.
  const Sophus::SE3d T_b_w = m_rig.T_c_b[0].inverse() * res.T_t_h * m_rig.T_c_b[0];
  traceCandidates(current, T_b_w, {res.affine});
  const int id = createKeyframe(current, T_b_w, {res.affine});
  m_frames.back() = {id, Sophus::SE3d()};
}

int Odometry::createKeyframe(const Pyramids& images, const Sophus::SE3d& T_b_w,
                             const std::vector<AffineBrightness>& affine) {
  const int index = m_frameCount - 1;
  const int id = m_window.addFrame(images, T_b_w, affine);
  m_keyframes[id] = {index, images, std::vector<std::vector<ImmaturePoint>>(m_rig.size())};
  m_keyframeFrameIndex[id] = index;
  if (multiCamera()) {
    // Several cameras: new candidates get a metric depth interval from the other cameras before activation.
    auto t = m_profile.scope("kf select+static");
    selectCandidates(id);
    traceStatic(id);
    if (m_window.frames().size() == 1) estimateStaticBrightness(id);
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
  if (m_settings.checkCalibration) m_window.logStaticDepthBias();
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
  if (!multiCamera()) {
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
  if (multiCamera()) {
    s.rhoMaxInit = 1.0 / m_settings.stereoMinDepth;
    s.maxSamples = m_settings.stereoMaxSamples;
    s.minQuality = m_settings.staticMinQuality;
  }
  return s;
}

void Odometry::selectCandidates(int keyframeId) {
  Keyframe& kf = m_keyframes.at(keyframeId);
  const TraceSettings settings = candidateSettings();
  for (int c = 0; c < m_rig.size(); ++c) {
    const ImageLevel& img = kf.images[c]->level(0);
    for (const auto& cand : m_selector.select(img, m_rig.cameras[c].maskImage()))
      if (auto p = ImmaturePoint::create(m_rig.cameras[c], img, cand.uv.cast<double>(), settings))
        kf.immature[c].push_back(std::move(*p));
  }
}

void Odometry::traceStatic(int keyframeId) {
  Keyframe& kf = m_keyframes.at(keyframeId);
  const std::vector<AffineBrightness>& affine = m_window.frame(keyframeId).params.affine;
  const TraceSettings settings = candidateSettings();
  for (int c = 0; c < m_rig.size(); ++c) {
    auto& points = kf.immature[c];
    parallelChunks(points.size(), [&](size_t, size_t begin, size_t end) {
      for (size_t i = begin; i < end; ++i) {
        for (int other = 0; other < m_rig.size(); ++other) {
          if (other == c) continue;
          HostTargetState state;
          state.T_t_h = m_rig.T_to_from(other, c);
          state.host = affine[c];
          state.target = affine[other];
          // A camera that cannot see the point must not undo a match in another camera.
          ImmaturePoint traced = points[i];
          if (traced.trace(m_rig.cameras[other], kf.images[other]->level(0), state, settings) !=
              TraceStatus::OutOfBounds)
            points[i] = std::move(traced);
        }
      }
    });
  }
}

void Odometry::estimateStaticBrightness(int keyframeId) {
  // Gain and offset of each camera relative to camera 0 from the matched patterns (least squares on
  // I_other = g * I_0 + b); later keyframes get theirs from tracking.
  const Keyframe& kf = m_keyframes.at(keyframeId);
  KeyframeParams params = m_window.frame(keyframeId).params;
  for (int other = 1; other < m_rig.size(); ++other) {
    const Sophus::SE3d T = m_rig.T_to_from(other, 0);
    const Camera& cam = m_rig.cameras[other];
    const ImageLevel& img = kf.images[other]->level(0);
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    int n = 0;
    for (const auto& p : kf.immature[0]) {
      if (p.lastStatus() != TraceStatus::Good) continue;
      for (int k = 0; k < kPatternSize; ++k) {
        Eigen::Vector2d uv;
        if (!projectBearing(p.pattern().bearings[k], p.rho(), T, cam, uv) || !cam.isInside(uv.x(), uv.y(), 1.0))
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
    const double det = n * sxx - sx * sx;
    if (n > 200 && det > 1e-9) {
      const double gain = (n * sxy - sx * sy) / det;
      if (gain > 0.5 && gain < 2.0) params.affine[other] = {std::log(gain), (sy - gain * sx) / n};
    }
  }
  m_window.setFrameParams(keyframeId, params);
}

void Odometry::traceCandidates(const Pyramids& images, const Sophus::SE3d& T_b_w,
                               const std::vector<AffineBrightness>& affine) {
  for (auto& [id, kf] : m_keyframes) {
    const WindowFrame& host = m_window.frame(id);
    for (int c = 0; c < m_rig.size(); ++c) {
      HostTargetState state;
      state.T_t_h = m_rig.T_c_b[c] * T_b_w * host.params.T_b_w.inverse() * m_rig.T_c_b[c].inverse();
      state.host = host.params.affine[c];
      state.target = affine[c];
      auto& points = kf.immature[c];
      parallelChunks(points.size(), [&](size_t, size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i)
          points[i].trace(m_rig.cameras[c], images[c]->level(0), state, m_settings.trace);
      });
      std::erase_if(points, [](const ImmaturePoint& p) {
        return p.lastStatus() == TraceStatus::OutOfBounds || p.numOutliers() > p.numGood() + 2;
      });
    }
  }
}

void Odometry::activateCandidates(int newKeyframeId) {
  const int nc = m_rig.size();
  const int cell = std::max(2, static_cast<int>(std::lround(m_activationCell)));
  std::vector<int> gridWidth(nc);
  std::vector<std::vector<char>> occupied(nc);
  std::vector<Sophus::SE3d> T_t_w(nc);
  for (int c = 0; c < nc; ++c) {
    const Camera& cam = m_rig.cameras[c];
    gridWidth[c] = (cam.width + cell - 1) / cell;
    occupied[c].assign(static_cast<size_t>(gridWidth[c]) * ((cam.height + cell - 1) / cell), 0);
    T_t_w[c] = m_window.cameraPose(newKeyframeId, c);
  }
  // Cells of all cameras of the new keyframe the point projects into (-1 where it does not).
  auto cellsOf = [&](const Sophus::SE3d& T_h_w, const Eigen::Vector3d& bearing, double rho) {
    std::vector<int> cells(nc, -1);
    for (int c = 0; c < nc; ++c) {
      Eigen::Vector2d uv;
      const Camera& cam = m_rig.cameras[c];
      if (projectBearing(bearing, rho, T_t_w[c] * T_h_w.inverse(), cam, uv) && cam.isInside(uv.x(), uv.y(), 0.0))
        cells[c] = static_cast<int>(uv.y()) / cell * gridWidth[c] + static_cast<int>(uv.x()) / cell;
    }
    return cells;
  };

  for (const auto& p : m_window.points()) {
    const auto cells = cellsOf(m_window.cameraPose(p.host, p.hostCam), p.bearing, p.rho);
    for (int c = 0; c < nc; ++c)
      if (cells[c] >= 0) occupied[c][cells[c]] = 1;
  }

  // The first keyframe has only the static traces between its cameras.
  const int minGood = m_window.frames().size() > 1 ? m_settings.activationMinGood : 1;
  // A cell taken in any camera rejects the candidate, so a surface seen by two cameras is hosted only once.
  for (auto& [id, kf] : m_keyframes) {
    if (id == newKeyframeId && !multiCamera()) continue;  // with several cameras its own candidates are matched
    for (int hc = 0; hc < nc; ++hc) {
      const Sophus::SE3d T_h_w = m_window.cameraPose(id, hc);
      std::erase_if(kf.immature[hc], [&](const ImmaturePoint& p) {
        if (p.lastStatus() != TraceStatus::Good || p.numGood() < minGood ||
            p.lastErrorPixels() > m_settings.activationMaxErrorPixels || p.rho() <= 0)
          return false;
        const auto cells = cellsOf(T_h_w, p.bearing(), p.rho());
        bool visible = false;
        for (int c = 0; c < nc; ++c) {
          if (cells[c] < 0) continue;
          if (occupied[c][cells[c]]) return false;
          visible = true;
        }
        if (!visible) return false;
        if (m_window.addPoint(id, p.pattern().uv, p.rho(), hc) < 0) return false;
        for (int c = 0; c < nc; ++c)
          if (cells[c] >= 0) occupied[c][cells[c]] = 1;
        return true;
      });
    }
  }

  const double ratio =
      static_cast<double>(m_window.points().size()) / (static_cast<double>(m_settings.targetActivePoints) * nc);
  m_activationCell = std::clamp(m_activationCell * std::sqrt(std::max(ratio, 0.25)), 3.0, 40.0);
}

void Odometry::removeOutlierPoints() {
  // Outlier residuals are already excluded from the optimisation. The good fraction only counts the host and newer
  // keyframes, the views tracing checked: new points are often occluded or strongly warped in older keyframes, and
  // dropping them for that starves the window (KITTI 00 frames 4000-4500: monocular scale; aiMotive up-ramp: a
  // 360° rig keeps keyframes from the level below, and the forward motion in the ramp tunnel stalls).
  // Those outliers instead mark the older keyframe for marginalisation, see marginalizeKeyframes.
  std::vector<int> remove;
  for (const auto& p : m_window.points()) {
    if (p.residuals.empty()) continue;
    int inside = 0, good = 0;
    const int hostFrame = m_keyframeFrameIndex.at(p.host);
    for (const auto& r : p.residuals) {
      if (m_keyframeFrameIndex.at(r.target) < hostFrame) continue;
      inside += r.state != ResidualState::OutOfBounds;
      good += r.state == ResidualState::Good;
    }
    if (p.numGood() == 0 || good < m_settings.pointMinGoodFraction * inside) remove.push_back(p.id);
  }
  for (int id : remove) m_window.removePoint(id);
}

void Odometry::marginalizeKeyframes() {
  {
    const auto& frames = m_window.frames();
    if (frames.size() <= 2) return;
    const int newest = frames.back().id;
    std::vector<int> flagged;
    for (size_t i = 0; i + 2 < frames.size(); ++i) {
      const int frameIndex = m_keyframeFrameIndex.at(frames[i].id);
      int hosted = 0, visible = 0, seen = 0, seenGood = 0;
      for (const auto& p : m_window.points()) {
        if (p.host == frames[i].id) {
          ++hosted;
          // A point counts once, however many cameras of the newest keyframe see it.
          visible += std::ranges::any_of(p.residuals, [&](const WindowResidual& r) {
            return r.target == newest && r.state == ResidualState::Good;
          });
        } else if (m_keyframeFrameIndex.at(p.host) > frameIndex) {
          for (const auto& r : p.residuals) {
            if (r.target != frames[i].id || r.state == ResidualState::OutOfBounds) continue;
            ++seen;
            seenGood += r.state == ResidualState::Good;
          }
        }
      }
      // DSO §3.1 only asks whether the newest keyframe still sees this keyframe's points, which a 360° rig answers
      // with its rear cameras long after the keyframe stopped seeing the scene ahead. So the reverse is asked too:
      // whether this keyframe sees the points of newer keyframes (by residuals that project inside its images).
      const bool stale = seen >= m_settings.marginalizeMinNewerResiduals &&
                         seenGood < m_settings.marginalizeNewerGoodFraction * seen;
      spdlog::trace("keyframe {} in window: {} of {} own points seen by the newest, {} of {} newer points good",
                    frameIndex, visible, hosted, seenGood, seen);
      if (stale || visible < m_settings.marginalizeVisibleFraction * std::max(hosted, 1)) flagged.push_back(frames[i].id);
    }
    for (int id : flagged) marginalize(id);
  }

  while (static_cast<int>(m_window.frames().size()) > m_settings.maxKeyframes) {
    const auto& frames = m_window.frames();
    const size_t n = frames.size();
    auto center = [&](size_t i) { return frames[i].params.T_b_w.inverse().translation(); };
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

MapPoint Odometry::mapPoint(const WindowPoint& p) const {
  const ImageLevel& img = m_window.frame(p.host).images[p.hostCam]->level(0);
  return {m_window.cameraPose(p.host, p.hostCam).inverse() * (p.bearing / p.rho),
          img.interpolateIntensity(static_cast<float>(p.pattern.uv.x()), static_cast<float>(p.pattern.uv.y())),
          m_keyframeFrameIndex.at(p.host), p.hostCam, p.pattern.uv, 1.0 / p.rho, p.numGood(),
          p.hRho > 0 ? 1.0 / (p.rho * std::sqrt(p.hRho)) : std::numeric_limits<double>::infinity()};
}

void Odometry::appendCandidatePoints(int keyframeId, std::vector<MapPoint>& out) const {
  if (!m_settings.mapCandidates) return;
  const Keyframe& kf = m_keyframes.at(keyframeId);
  for (int c = 0; c < m_rig.size(); ++c) {
    const Sophus::SE3d T_w_c = m_window.cameraPose(keyframeId, c).inverse();
    const ImageLevel& img = kf.images[c]->level(0);
    for (const auto& p : kf.immature[c]) {
      const double rho = p.rho();
      if (p.numGood() < m_settings.candidateMinGood || rho <= kMinMapRho || rho < p.rhoMin() || rho > p.rhoMax())
        continue;
      const double interval = 0.5 * (p.rhoMax() - p.rhoMin()) / rho;
      if (interval > m_settings.candidateMaxInterval) continue;
      const Eigen::Vector2d& uv = p.pattern().uv;
      out.push_back({T_w_c * (p.bearing() / rho),
                     img.interpolateIntensity(static_cast<float>(uv.x()), static_cast<float>(uv.y())), kf.frameIndex,
                     c, uv, 1.0 / rho, p.numGood(), interval, MapPointSource::Candidate});
    }
  }
}

KeyframeRecord Odometry::makeRecord(int keyframeId) const {
  const Keyframe& kf = m_keyframes.at(keyframeId);
  KeyframeRecord rec{kf.frameIndex, m_window.frame(keyframeId).params.T_b_w.inverse(), {},
                     m_window.frame(keyframeId).params.affine};
  for (int c = 0; c < m_rig.size(); ++c) {
    const Camera& cam = m_rig.cameras[c];
    CameraFeatures f = extractFeatures(cam, toGray8(kf.images[c]->level(0)), m_settings.features);
    // Depth from all window points that project into this camera, plus its own converged candidates.
    const Sophus::SE3d T_c_w = m_window.cameraPose(keyframeId, c);
    std::vector<Eigen::Vector2f> uv;
    std::vector<float> rho;
    auto addPoint = [&](const Sophus::SE3d& T_c_h, const Eigen::Vector3d& bearing, double r) {
      const Eigen::Vector3d x = T_c_h.so3() * bearing + r * T_c_h.translation();
      Eigen::Vector2d q;
      if (x.norm() < 1e-9 || !cam.project(x, q) || !cam.isInside(q.x(), q.y(), 0.0)) return;
      uv.push_back(q.cast<float>());
      rho.push_back(static_cast<float>(r / x.norm()));
    };
    for (const auto& p : m_window.points())
      if (p.numGood() > 0 && p.rho > kMinMapRho)
        addPoint(T_c_w * m_window.cameraPose(p.host, p.hostCam).inverse(), p.bearing, p.rho);
    for (const auto& p : kf.immature[c])
      if (p.numGood() >= m_settings.candidateMinGood && p.rho() > kMinMapRho &&
          0.5 * (p.rhoMax() - p.rhoMin()) / p.rho() <= m_settings.candidateMaxInterval)
        addPoint(Sophus::SE3d(), p.bearing(), p.rho());
    assignDepth(f, uv, rho, m_settings.features);
    for (const auto& p : m_window.points())
      if (p.host == keyframeId && p.hostCam == c && p.numGood() > 0 && p.rho > kMinMapRho) {
        f.pointUv.push_back(p.pattern.uv.cast<float>());
        f.pointRho.push_back(static_cast<float>(p.rho));
      }
    rec.cameras.push_back(std::move(f));
  }
  return rec;
}

std::vector<KeyframeRecord> Odometry::keyframeRecords() const {
  std::vector<KeyframeRecord> out = m_records;
  if (m_settings.extractFeatures)
    for (const auto& [id, kf] : m_keyframes) out.push_back(makeRecord(id));
  std::sort(out.begin(), out.end(), [](const KeyframeRecord& a, const KeyframeRecord& b) { return a.frameIndex < b.frameIndex; });
  return out;
}

void Odometry::marginalize(int keyframeId) {
  if (m_settings.extractFeatures) {
    auto t = m_profile.scope("kf features");
    m_records.push_back(makeRecord(keyframeId));
  }
  for (const auto& p : m_window.points())
    if (p.host == keyframeId && p.numGood() > 0 && p.rho > kMinMapRho) m_marginalizedPoints.push_back(mapPoint(p));
  appendCandidatePoints(keyframeId, m_marginalizedPoints);
  m_keyframePoses[keyframeId] = m_window.frame(keyframeId).params.T_b_w;
  m_keyframeAffine[keyframeId] = m_window.frame(keyframeId).params.affine;
  m_window.marginalizeFrame(keyframeId);
  m_keyframes.erase(keyframeId);
}

void Odometry::buildReference() {
  const WindowFrame& kf = m_window.frame(m_referenceId);
  m_reference.clear();
  for (int c = 0; c < m_rig.size(); ++c) {
    const Camera& cam = m_rig.cameras[c];
    const Sophus::SE3d T_c_w = m_window.cameraPose(kf.id, c);
    std::vector<Eigen::Vector2i> pixels;
    std::vector<double> rhos;
    for (const auto& p : m_window.points()) {
      if (p.host != kf.id && p.numGood() == 0) continue;
      const Sophus::SE3d T_c_h = T_c_w * m_window.cameraPose(p.host, p.hostCam).inverse();
      const Eigen::Vector3d x = T_c_h.so3() * p.bearing + p.rho * T_c_h.translation();
      Eigen::Vector2d uv;
      if (!cam.project(x, uv) || !cam.isInside(uv.x(), uv.y(), 2.0)) continue;
      pixels.emplace_back(static_cast<int>(std::lround(uv.x())), static_cast<int>(std::lround(uv.y())));
      rhos.push_back(p.rho / x.norm());
    }
    m_reference.emplace_back(cam, *kf.images[c], pixels, rhos, kf.params.affine[c], m_settings.tracking.gradientWeightC);
  }
}

double Odometry::translationFlow(const Sophus::SE3d& T_f_ref) const {
  double sum = 0;
  int n = 0;
  Eigen::Vector2d full, rot;
  for (int c = 0; c < m_rig.size(); ++c) {
    const Sophus::SE3d T = m_rig.T_c_b[c] * T_f_ref * m_rig.T_c_b[c].inverse();
    const Camera& cam = m_rig.cameras[c];
    const auto& pts = m_reference[c].points(0);
    for (size_t i = 0; i < pts.size(); i += 4) {
      const Eigen::Vector3d r = T.so3() * pts[i].bearing;
      if (!cam.project(Eigen::Vector3d(r + pts[i].rho * T.translation()), full) || !cam.project(r, rot)) continue;
      sum += (full - rot).norm();
      ++n;
    }
  }
  return n > 0 ? sum / n : 0.0;
}

bool Odometry::needKeyframe(const TrackingResult& res) const {
  double brightness = 0;
  const auto& ref = m_window.frame(m_referenceId).params.affine;
  for (int c = 0; c < m_rig.size(); ++c) brightness = std::max(brightness, std::abs(res.affine[c].a - ref[c].a));
  const double score = res.meanFlow / m_settings.kfFlow + translationFlow(res.T_t_h) / m_settings.kfTranslationFlow +
                       brightness / m_settings.kfBrightness;
  return score > 1.0 || (m_referenceRmse > 0 && res.rmse > m_settings.kfRmseFactor * m_referenceRmse);
}

void Odometry::storeKeyframePoses() {
  for (const auto& f : m_window.frames()) m_keyframePoses[f.id] = f.params.T_b_w;
}

std::vector<std::optional<Sophus::SE3d>> Odometry::poses() const {
  std::map<int, Sophus::SE3d> current = m_keyframePoses;
  for (const auto& f : m_window.frames()) current[f.id] = f.params.T_b_w;
  std::vector<std::optional<Sophus::SE3d>> out;
  for (const auto& r : m_frames)
    out.push_back(r.keyframe < 0 ? std::nullopt : std::optional((r.T_f_kf * current.at(r.keyframe)).inverse()));
  return out;
}

std::vector<MapPoint> Odometry::mapPoints() const {
  std::vector<MapPoint> out = m_marginalizedPoints;
  for (const auto& p : m_window.points())
    if (p.numGood() > 0 && p.rho > kMinMapRho) out.push_back(mapPoint(p));
  for (const auto& [id, kf] : m_keyframes) appendCandidatePoints(id, out);
  return out;
}

std::vector<std::vector<AffineBrightness>> Odometry::brightness() const {
  std::map<int, std::vector<AffineBrightness>> keyframe = m_keyframeAffine;
  for (const auto& f : m_window.frames()) keyframe[f.id] = f.params.affine;
  std::vector<std::vector<AffineBrightness>> out;
  for (const auto& r : m_frames) {
    if (r.keyframe < 0) out.emplace_back();
    else if (r.affine.empty()) out.push_back(keyframe.at(r.keyframe));
    else out.push_back(r.affine);
  }
  return out;
}

std::vector<int> Odometry::keyframeIndices() const {
  std::vector<int> out;
  for (const auto& [id, index] : m_keyframeFrameIndex) out.push_back(index);
  return out;
}

std::vector<int> Odometry::windowKeyframeIndices() const {
  std::vector<int> out;
  for (const auto& [id, kf] : m_keyframes) out.push_back(kf.frameIndex);
  return out;
}

}  // namespace sdv
