#pragma once

#include <memory>
#include <optional>
#include <vector>

#include <Eigen/Core>
#include <sophus/se3.hpp>

#include <sdv/keyframe_features.h>
#include <sdv/place_database.h>
#include <sdv/rig.h>

namespace sdv {

struct LoopSettings {
  int candidates = 3;  // best database matches verified per query
  double minNormalizedScore = 0.3;  // candidate score / score against the previous keyframe
  int minGapFrames = 150;  // candidates are at least this many input frames older
  double ratio = 0.8;  // descriptor ratio test
  int maxHamming = 64;
  int ransacIterations = 400;
  double inlierPixels = 3.0;  // angular reprojection threshold, in pixels of the observing camera
  int minInliers = 40;
  // Odometry consistency: the loop may correct the odometry relative pose by at most
  // base + rate * (path length between the two keyframes); vertical separately (multi-storey buildings).
  double driftBaseMeters = 1.0, driftRate = 0.03;
  double verticalBaseMeters = 0.5, verticalRate = 0.01;
  double driftBaseDeg = 5.0, driftRateDegPerMeter = 0.02;
  Eigen::Vector3d up = Eigen::Vector3d::UnitZ();  // world vertical (vehicle rigs: z; KITTI camera frame: -y)
  // Temporal consistency: loops of nearby query keyframes must imply the same pose correction.
  int temporalWindow = 5;  // query keyframes on either side
  int minConsistent = 2;  // other loops that agree
  double consistencyMeters = 1.0, consistencyDeg = 3.0;
};

struct LoopConstraint {
  int query, match;  // record indices, match older than query
  Sophus::SE3d T_q_m;  // body of match -> body of query
  int inliers;
  double rmsePixels;
  double score;  // normalised BoW score
};

struct LoopStats {
  long long queries = 0, candidates = 0, fewMatches = 0, fewInliers = 0, inconsistentOdometry = 0, verified = 0;
};

// Geometric verification of one candidate pair: ORB matches (every query camera against all match cameras),
// RANSAC on matches with depth on both sides (3-point absolute orientation, Horn 1987 / Umeyama 1991) scored
// with the angular reprojection error of every match with depth on at least one side, then a robust refinement.
std::optional<LoopConstraint> verifyLoop(const Rig& rig, const KeyframeRecord& query, const KeyframeRecord& match,
                                         const LoopSettings& settings, LoopStats* stats = nullptr);

// Keyframes in input order; each is checked against the earlier ones when it is added.
class LoopDetector {
 public:
  LoopDetector(const Rig& rig, std::shared_ptr<const Vocabulary> vocabulary, LoopSettings settings = {});

  // Loops of this keyframe that pass geometric verification and the odometry-consistency check.
  std::vector<LoopConstraint> addKeyframe(const KeyframeRecord& record);
  const std::vector<KeyframeRecord>& records() const { return m_records; }
  const LoopStats& stats() const { return m_stats; }
  // Odometry correction a loop implies for its query keyframe, in the world frame: T_w_q' = C * T_w_q.
  Sophus::SE3d correction(const LoopConstraint& loop) const;
  // Loops confirmed by loops of neighbouring query keyframes (all loops known so far).
  std::vector<LoopConstraint> temporallyConsistent(const std::vector<LoopConstraint>& loops) const;

 private:
  bool consistentWithOdometry(const LoopConstraint& loop) const;

  Rig m_rig;
  std::shared_ptr<const Vocabulary> m_vocabulary;
  LoopSettings m_settings;
  PlaceDatabase m_db;  // image = record * cameras + camera
  std::vector<KeyframeRecord> m_records;
  std::vector<double> m_pathLength;  // odometry distance from the first keyframe
  LoopStats m_stats;
};

}  // namespace sdv
