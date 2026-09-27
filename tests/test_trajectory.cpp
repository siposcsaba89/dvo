#include <random>

#include <gtest/gtest.h>

#include <sdv/eval/trajectory.h>

namespace {

std::vector<Sophus::SE3d> makeTrajectory(int n) {
  std::vector<Sophus::SE3d> poses;
  for (int i = 0; i < n; ++i) {
    const double t = i * 0.1;
    poses.emplace_back(Sophus::SO3d::rotY(0.05 * t) * Sophus::SO3d::rotX(0.01 * std::sin(t)),
                       Eigen::Vector3d(std::sin(0.2 * t) * 10.0, 0.1 * t, t * 5.0));
  }
  return poses;
}

}  // namespace

TEST(Trajectory, RecoversSimilarity) {
  const auto gt = makeTrajectory(200);
  const Sophus::SO3d rot = Sophus::SO3d::exp(Eigen::Vector3d(0.3, -0.2, 0.9));
  const double scale = 0.37;
  const Eigen::Vector3d trans(4, -2, 7);
  std::vector<Sophus::SE3d> est;
  for (const auto& p : gt) {
    Eigen::Vector3d c = rot.inverse() * (p.translation() - trans) / scale;
    est.emplace_back(rot.inverse() * p.so3(), c);
  }

  const auto ate = sdv::absoluteTrajectoryError(gt, est, true);
  EXPECT_NEAR(ate.alignment.scale, scale, 1e-9);
  EXPECT_LT(ate.rmse, 1e-9);

  for (auto& p : est) p = ate.alignment.applyToPose(p);
  const auto seg = sdv::segmentDriftError(gt, est);
  EXPECT_LT(seg.translationPercent, 1e-6);
  EXPECT_LT(seg.rotationDegPer100m, 1e-6);
}

TEST(Trajectory, RigidAlignmentKeepsScale) {
  const auto gt = makeTrajectory(100);
  std::vector<Sophus::SE3d> est;
  for (const auto& p : gt) est.emplace_back(p.so3(), p.translation() * 2.0);
  const auto ate = sdv::absoluteTrajectoryError(gt, est, false);
  EXPECT_DOUBLE_EQ(ate.alignment.scale, 1.0);
  EXPECT_GT(ate.rmse, 1.0);
}

TEST(Trajectory, DriftDetectsScaleError) {
  const auto gt = makeTrajectory(2000);
  std::vector<Sophus::SE3d> est;
  for (const auto& p : gt) est.emplace_back(p.so3(), p.translation() * 1.01);
  const auto seg = sdv::segmentDriftError(gt, est);
  ASSERT_GT(seg.numSegments, 0);
  EXPECT_NEAR(seg.translationPercent, 1.0, 0.05);
}

TEST(Trajectory, RelativePoseErrorMeasuresJitter) {
  const auto gt = makeTrajectory(200);
  EXPECT_NEAR(sdv::relativePoseError(gt, gt).rotationRmseDeg, 0.0, 1e-9);
  std::vector<Sophus::SE3d> est = gt;
  // Every other pose rotated by 0.1 deg: each consecutive pair is off by 0.1 deg.
  for (size_t i = 1; i < est.size(); i += 2)
    est[i] = est[i] * Sophus::SE3d(Sophus::SO3d::exp(Eigen::Vector3d(0, 0.1 * M_PI / 180.0, 0)), Eigen::Vector3d::Zero());
  const auto rpe = sdv::relativePoseError(gt, est);
  EXPECT_EQ(rpe.numPairs, 199);
  EXPECT_NEAR(rpe.rotationRmseDeg, 0.1, 1e-6);
  // A pure scale difference vanishes with the right scale.
  std::vector<Sophus::SE3d> scaled;
  for (const auto& p : gt) scaled.emplace_back(p.so3(), p.translation() * 0.5);
  EXPECT_NEAR(sdv::relativePoseError(gt, scaled, 1, 2.0).translationRmse, 0.0, 1e-9);
  EXPECT_GT(sdv::relativePoseError(gt, scaled).translationRmse, 0.1);
}
