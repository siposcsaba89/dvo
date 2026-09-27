#include <sdv/loop_detector.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <random>

#include <ceres/ceres.h>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <sophus/ceres_typetraits.hpp>

namespace sdv {

namespace {

struct Correspondence {
  int qc, mc;
  Eigen::Vector3d bq, bm;  // bearings in the query / match camera
  bool hasXq, hasXm;
  Eigen::Vector3d Xq, Xm;  // points in the query / match body frame
};

double angle(const Eigen::Vector3d& a, const Eigen::Vector3d& b) { return std::atan2(a.cross(b).norm(), a.dot(b)); }

Eigen::Vector3d bodyPoint(const Rig& rig, int cam, const Eigen::Vector3f& bearing, float rho) {
  return rig.T_c_b[cam].inverse() * (bearing.cast<double>() / rho);
}

std::vector<Correspondence> matchFeatures(const Rig& rig, const KeyframeRecord& q, const KeyframeRecord& m,
                                          const LoopSettings& settings) {
  const int nc = rig.size();
  cv::Mat all;
  std::vector<std::pair<int, int>> source;  // (camera, feature) of each row of `all`
  for (int c = 0; c < nc; ++c) {
    const CameraFeatures& f = m.cameras[c];
    if (f.descriptors.empty()) continue;
    all.push_back(f.descriptors);
    for (size_t i = 0; i < f.size(); ++i) source.emplace_back(c, static_cast<int>(i));
  }
  if (all.rows < 2) return {};
  cv::BFMatcher matcher(cv::NORM_HAMMING);
  // Best query feature per match feature, so that one match feature is used once.
  std::map<int, std::pair<float, Correspondence>> best;
  for (int qc = 0; qc < nc; ++qc) {
    const CameraFeatures& fq = q.cameras[qc];
    if (fq.descriptors.empty()) continue;
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(fq.descriptors, all, knn, 2);
    for (const auto& k : knn) {
      if (k.size() < 2 || k[0].distance > settings.maxHamming || k[0].distance > settings.ratio * k[1].distance) continue;
      const auto [mc, j] = source[k[0].trainIdx];
      const int i = k[0].queryIdx;
      const CameraFeatures& fm = m.cameras[mc];
      Correspondence c{qc, mc, fq.bearings[i].cast<double>(), fm.bearings[j].cast<double>(), fq.rho[i] > 0,
                       fm.rho[j] > 0, {}, {}};
      if (c.hasXq) c.Xq = bodyPoint(rig, qc, fq.bearings[i], fq.rho[i]);
      if (c.hasXm) c.Xm = bodyPoint(rig, mc, fm.bearings[j], fm.rho[j]);
      if (!c.hasXq && !c.hasXm) continue;
      auto it = best.find(k[0].trainIdx);
      if (it == best.end() || k[0].distance < it->second.first) best[k[0].trainIdx] = {k[0].distance, c};
    }
  }
  std::vector<Correspondence> out;
  for (auto& [key, v] : best) out.push_back(v.second);
  return out;
}

// Reprojection error in pixels of the observing camera(s); the worse one when both sides have depth.
double errorPixels(const Rig& rig, const Correspondence& c, const Sophus::SE3d& T_q_m) {
  double e = 0;
  if (c.hasXm) e = std::max(e, angle(rig.T_c_b[c.qc] * (T_q_m * c.Xm), c.bq) * rig.cameras[c.qc].fx);
  if (c.hasXq) e = std::max(e, angle(rig.T_c_b[c.mc] * (T_q_m.inverse() * c.Xq), c.bm) * rig.cameras[c.mc].fx);
  return e;
}

std::vector<int> inliersOf(const Rig& rig, const std::vector<Correspondence>& corr, const Sophus::SE3d& T,
                           double threshold) {
  std::vector<int> in;
  for (size_t i = 0; i < corr.size(); ++i)
    if (errorPixels(rig, corr[i], T) < threshold) in.push_back(static_cast<int>(i));
  return in;
}

// Unit bearing residual of a point seen by a camera: pose T = exp(delta) * T0 (T0 = T_q_m), point in the match
// body (seen from the query) or in the query body (seen from the match, through T^-1).
struct BearingResidual {
  Eigen::Vector3d X, bearing;
  Sophus::SE3d T_c_b, T0;
  bool inverse;

  template <typename T>
  bool operator()(const T* delta, T* residual) const {
    const Sophus::SE3<T> pose = Sophus::SE3<T>::exp(Eigen::Map<const Eigen::Matrix<T, 6, 1>>(delta)) * T0.cast<T>();
    const Eigen::Matrix<T, 3, 1> x = inverse ? (pose.inverse() * X.cast<T>()).eval() : (pose * X.cast<T>()).eval();
    const Eigen::Matrix<T, 3, 1> p = T_c_b.cast<T>() * x;
    const Eigen::Matrix<T, 3, 1> r = p / p.norm() - bearing.cast<T>();
    for (int k = 0; k < 3; ++k) residual[k] = r[k];
    return true;
  }
};

Sophus::SE3d refine(const Rig& rig, const std::vector<Correspondence>& corr, const std::vector<int>& inliers,
                    const Sophus::SE3d& T0, double thresholdRad) {
  double delta[6] = {0, 0, 0, 0, 0, 0};
  ceres::Problem problem;
  auto add = [&](const Eigen::Vector3d& X, const Eigen::Vector3d& b, int cam, bool inverse) {
    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<BearingResidual, 3, 6>(
                                 new BearingResidual{X, b, rig.T_c_b[cam], T0, inverse}),
                             new ceres::HuberLoss(thresholdRad), delta);
  };
  for (int i : inliers) {
    const Correspondence& c = corr[i];
    if (c.hasXm) add(c.Xm, c.bq, c.qc, false);
    if (c.hasXq) add(c.Xq, c.bm, c.mc, true);
  }
  ceres::Solver::Options options;
  options.max_num_iterations = 20;
  options.linear_solver_type = ceres::DENSE_QR;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  return Sophus::SE3d::exp(Eigen::Map<const Sophus::Vector6d>(delta)) * T0;
}

}  // namespace

std::optional<LoopConstraint> verifyLoop(const Rig& rig, const KeyframeRecord& query, const KeyframeRecord& match,
                                         const LoopSettings& settings, LoopStats* stats) {
  LoopStats dummy;
  LoopStats& st = stats ? *stats : dummy;
  const std::vector<Correspondence> corr = matchFeatures(rig, query, match, settings);
  std::vector<int> both;
  for (size_t i = 0; i < corr.size(); ++i)
    if (corr[i].hasXq && corr[i].hasXm) both.push_back(static_cast<int>(i));
  if (static_cast<int>(corr.size()) < settings.minInliers || both.size() < 6) {
    ++st.fewMatches;
    return std::nullopt;
  }

  std::mt19937 rng(static_cast<unsigned>(query.frameIndex * 7919 + match.frameIndex));
  std::uniform_int_distribution<size_t> pick(0, both.size() - 1);
  Sophus::SE3d bestT;
  size_t bestCount = 0;
  for (int it = 0; it < settings.ransacIterations; ++it) {
    const int a = both[pick(rng)], b = both[pick(rng)], c = both[pick(rng)];
    if (a == b || b == c || a == c) continue;
    Eigen::Matrix3d src, dst;
    src << corr[a].Xm, corr[b].Xm, corr[c].Xm;
    dst << corr[a].Xq, corr[b].Xq, corr[c].Xq;
    if ((src.col(1) - src.col(0)).cross(src.col(2) - src.col(0)).norm() < 1e-2) continue;
    const Eigen::Matrix4d M = Eigen::umeyama(src, dst, false);
    const Sophus::SE3d T(Eigen::Quaterniond(Eigen::Matrix3d(M.topLeftCorner<3, 3>())).normalized(),
                         M.topRightCorner<3, 1>());
    size_t count = 0;
    for (const auto& co : corr) count += errorPixels(rig, co, T) < settings.inlierPixels;
    if (count > bestCount) bestCount = count, bestT = T;
  }
  if (static_cast<int>(bestCount) < settings.minInliers / 2) {
    ++st.fewInliers;
    return std::nullopt;
  }
  const double thresholdRad = settings.inlierPixels / rig.cameras[0].fx;
  Sophus::SE3d T = bestT;
  std::vector<int> inliers = inliersOf(rig, corr, T, settings.inlierPixels);
  for (int round = 0; round < 2; ++round) {
    T = refine(rig, corr, inliers, T, thresholdRad);
    inliers = inliersOf(rig, corr, T, settings.inlierPixels);
  }
  if (static_cast<int>(inliers.size()) < settings.minInliers) {
    ++st.fewInliers;
    return std::nullopt;
  }
  double sq = 0;
  for (int i : inliers) sq += std::pow(errorPixels(rig, corr[i], T), 2);
  return LoopConstraint{-1, -1, T, static_cast<int>(inliers.size()), std::sqrt(sq / inliers.size()), 0.0};
}

LoopDetector::LoopDetector(const Rig& rig, std::shared_ptr<const Vocabulary> vocabulary, LoopSettings settings)
    : m_rig(rig), m_vocabulary(std::move(vocabulary)), m_settings(std::move(settings)) {}

std::vector<LoopConstraint> LoopDetector::addKeyframe(const KeyframeRecord& record) {
  const int nc = m_rig.size();
  const int q = static_cast<int>(m_records.size());
  m_pathLength.push_back(q == 0 ? 0.0
                                : m_pathLength.back() +
                                      (record.T_w_b.translation() - m_records.back().T_w_b.translation()).norm());
  m_records.push_back(record);
  std::vector<BowVector> bows;
  for (const auto& f : record.cameras) bows.push_back(m_vocabulary->transform(f.descriptors));

  std::vector<LoopConstraint> out;
  if (q > 0) {
    ++m_stats.queries;
    // Normaliser (Galvez-Lopez & Tardos 2012): similarity of this place to itself a moment ago.
    double self = 0;
    for (int c = 0; c < nc; ++c)
      self = std::max(self, PlaceDatabase::score(bows[c], m_db.vector((q - 1) * nc + c)));
    auto eligible = [&](int image) {
      return m_records[image / nc].frameIndex + m_settings.minGapFrames <= record.frameIndex;
    };
    std::map<int, double> score;  // candidate record -> best score over camera pairs
    for (int c = 0; c < nc; ++c)
      for (const auto& m : m_db.query(bows[c], static_cast<size_t>(m_settings.candidates * nc), eligible))
        score[m.image / nc] = std::max(score[m.image / nc], m.score);
    std::vector<std::pair<double, int>> ranked;
    for (const auto& [r, s] : score) ranked.emplace_back(self > 0 ? s / self : s, r);
    std::sort(ranked.rbegin(), ranked.rend());
    for (size_t k = 0; k < ranked.size() && k < static_cast<size_t>(m_settings.candidates); ++k) {
      if (ranked[k].first < m_settings.minNormalizedScore) break;
      ++m_stats.candidates;
      auto loop = verifyLoop(m_rig, record, m_records[ranked[k].second], m_settings, &m_stats);
      if (!loop) continue;
      loop->query = q;
      loop->match = ranked[k].second;
      loop->score = ranked[k].first;
      if (!consistentWithOdometry(*loop)) {
        ++m_stats.inconsistentOdometry;
        continue;
      }
      ++m_stats.verified;
      out.push_back(*loop);
    }
  }
  for (auto& b : bows) m_db.add(std::move(b));
  return out;
}

Sophus::SE3d LoopDetector::correction(const LoopConstraint& loop) const {
  return m_records[loop.match].T_w_b * loop.T_q_m.inverse() * m_records[loop.query].T_w_b.inverse();
}

bool LoopDetector::consistentWithOdometry(const LoopConstraint& loop) const {
  const double path = m_pathLength[loop.query] - m_pathLength[loop.match];
  const Sophus::SE3d corrected = m_records[loop.match].T_w_b * loop.T_q_m.inverse();
  const Sophus::SE3d& odometry = m_records[loop.query].T_w_b;
  const Eigen::Vector3d d = corrected.translation() - odometry.translation();
  const double vertical = std::abs(d.dot(m_settings.up));
  const double horizontal = (d - d.dot(m_settings.up) * m_settings.up).norm();
  const double rotationDeg = (corrected.so3() * odometry.so3().inverse()).log().norm() * 180.0 / M_PI;
  return horizontal <= m_settings.driftBaseMeters + m_settings.driftRate * path &&
         vertical <= m_settings.verticalBaseMeters + m_settings.verticalRate * path &&
         rotationDeg <= m_settings.driftBaseDeg + m_settings.driftRateDegPerMeter * path;
}

std::vector<LoopConstraint> LoopDetector::temporallyConsistent(const std::vector<LoopConstraint>& loops) const {
  std::vector<Sophus::SE3d> corrections;
  for (const auto& l : loops) corrections.push_back(correction(l));
  std::vector<LoopConstraint> out;
  for (size_t i = 0; i < loops.size(); ++i) {
    const Sophus::SE3d& T_w_q = m_records[loops[i].query].T_w_b;
    const Sophus::SE3d a = corrections[i] * T_w_q;
    std::vector<int> agreeing;
    for (size_t j = 0; j < loops.size(); ++j) {
      if (loops[j].query == loops[i].query || std::abs(loops[j].query - loops[i].query) > m_settings.temporalWindow)
        continue;
      if (std::find(agreeing.begin(), agreeing.end(), loops[j].query) != agreeing.end()) continue;
      const Sophus::SE3d b = corrections[j] * T_w_q;
      if ((a.translation() - b.translation()).norm() <= m_settings.consistencyMeters &&
          (a.so3() * b.so3().inverse()).log().norm() * 180.0 / M_PI <= m_settings.consistencyDeg)
        agreeing.push_back(loops[j].query);
    }
    if (static_cast<int>(agreeing.size()) >= m_settings.minConsistent) out.push_back(loops[i]);
  }
  return out;
}

}  // namespace sdv
