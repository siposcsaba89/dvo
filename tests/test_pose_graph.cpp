#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include <sdv/pose_graph.h>

namespace {

// Closed square drive, 4 x 25 m, keyframes every metre; odometry with a constant yaw bias per keyframe (0.06 deg/m,
// ten times the real rigs).
struct Drive {
  std::vector<Sophus::SE3d> truth, odometry;
};

Drive squareDrive(double yawBiasPerKeyframe) {
  Drive d;
  Sophus::SE3d T, O;
  for (int side = 0; side < 4; ++side)
    for (int k = 0; k < 25; ++k) {
      d.truth.push_back(T);
      d.odometry.push_back(O);
      const double yaw = k == 24 ? M_PI / 2 : 0.0;
      const Sophus::SE3d step(Sophus::SO3d::rotZ(yaw), Eigen::Vector3d(1, 0, 0));
      T = T * step;
      O = O * step * Sophus::SE3d(Sophus::SO3d::rotZ(yawBiasPerKeyframe), Eigen::Vector3d::Zero());
    }
  d.truth.push_back(T);
  d.odometry.push_back(O);
  return d;
}

sdv::LoopConstraint trueLoop(const Drive& d, int query, int match) {
  return {query, match, d.truth[query].inverse() * d.truth[match], 100, 1.0, 1.0};
}

double maxError(const std::vector<Sophus::SE3d>& a, const std::vector<Sophus::SE3d>& b) {
  double e = 0;
  for (size_t i = 0; i < a.size(); ++i) e = std::max(e, (a[i].translation() - b[i].translation()).norm());
  return e;
}

}  // namespace

TEST(PoseGraph, LoopRemovesDrift) {
  const Drive d = squareDrive(0.001);
  const int last = static_cast<int>(d.truth.size()) - 1;
  const double before = maxError(d.odometry, d.truth);
  ASSERT_GT(before, 1.0);
  const auto result = sdv::optimizePoseGraph(d.odometry, {trueLoop(d, last, 0), trueLoop(d, last - 1, 1)});
  EXPECT_LT((result.T_w_b[last].translation() - d.truth[last].translation()).norm(), 0.1);
  EXPECT_LT(maxError(result.T_w_b, d.truth), 0.3 * before);
  EXPECT_EQ(result.loopsRejected, 0);
  EXPECT_LT(result.finalCost, result.initialCost);
}

TEST(PoseGraph, FalseLoopIsRejected) {
  const Drive d = squareDrive(0.001);
  const int last = static_cast<int>(d.truth.size()) - 1;
  std::vector<sdv::LoopConstraint> loops;
  for (int k = 0; k < 4; ++k) loops.push_back(trueLoop(d, last - k, k));
  // A wrong loop half way round: claims keyframe 50 is where keyframe 5 was.
  loops.push_back({50, 5, Sophus::SE3d(), 60, 1.0, 1.0});
  const auto result = sdv::optimizePoseGraph(d.odometry, loops);
  EXPECT_EQ(result.loopsRejected, 1);
  EXPECT_LT(maxError(result.T_w_b, d.truth), 0.4);
}

TEST(PoseGraph, CorrectionInterpolatesBetweenKeyframes) {
  const std::vector<int> frames = {0, 10};
  const std::vector<Sophus::SE3d> before = {Sophus::SE3d(), Sophus::SE3d()};
  const std::vector<Sophus::SE3d> after = {Sophus::SE3d(), Sophus::SE3d(Sophus::SO3d::rotZ(0.1), Eigen::Vector3d(1, 0, 0))};
  const sdv::PoseCorrection c(frames, before, after);
  EXPECT_LT((c.at(10).log() - after[1].log()).norm(), 1e-12);
  EXPECT_LT(c.at(0).log().norm(), 1e-12);
  EXPECT_LT((c.at(5).log() - 0.5 * after[1].log()).norm(), 1e-9);
  EXPECT_LT((c.at(20).log() - after[1].log()).norm(), 1e-12);
}
