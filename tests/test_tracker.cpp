#include <cmath>

#include <gtest/gtest.h>

#include <sdv/tracker.h>

#include <synthetic_scene.h>

namespace {

constexpr int kW = 320, kH = 240, kLevels = 4;

sdv::ReferenceFrame makeReference(const sdv::Camera& cam, const sdv::ImagePyramid& pyr) {
  std::vector<Eigen::Vector2i> pixels;
  std::vector<double> rho;
  for (int v = 4; v < kH - 4; v += 3)
    for (int u = 4; u < kW - 4; u += 3) {
      Eigen::Vector3d b;
      if (!cam.unproject(Eigen::Vector2d(u, v), b) || synthetic::trueRho(b) <= 0) continue;
      pixels.emplace_back(u, v);
      rho.push_back(synthetic::trueRho(b));
    }
  return sdv::ReferenceFrame(cam, pyr, pixels, rho, {}, 50.0);
}

class TrackerTest : public ::testing::TestWithParam<sdv::Camera> {};

}  // namespace

TEST_P(TrackerTest, RecoversMotionAndBrightness) {
  const sdv::Camera& cam = GetParam();
  const Sophus::SE3d T_t_h =
      Sophus::SE3d::exp((Sophus::Vector6d() << 0.08, -0.03, -0.25, 0.01, -0.015, 0.008).finished());
  const double gain = 1.15, offset = -6.0;

  const sdv::ImagePyramid host(synthetic::renderHost(kW, kH), kLevels);
  const sdv::ImagePyramid target(synthetic::renderTarget(cam, T_t_h, gain, offset), kLevels);
  const sdv::ReferenceFrame ref = makeReference(cam, host);

  const sdv::FrameTracker tracker(sdv::TrackingSettings{});
  const auto res = tracker.track(ref, target, {Sophus::SE3d()}, {});
  ASSERT_TRUE(res.ok);
  const Sophus::Vector6d err = (res.T_t_h * T_t_h.inverse()).log();
  EXPECT_LT(err.head<3>().norm(), 2e-3);
  EXPECT_LT(err.tail<3>().norm(), 5e-4);
  EXPECT_NEAR(std::exp(res.affine.a), gain, 0.01);
  EXPECT_NEAR(res.affine.b, offset, 1.0);
  EXPECT_GT(res.inlierRatio, 0.9);
}

TEST_P(TrackerTest, PicksBestHypothesis) {
  const sdv::Camera& cam = GetParam();
  const Sophus::SE3d T_t_h =
      Sophus::SE3d::exp((Sophus::Vector6d() << 0.02, 0.0, -0.4, 0.0, 0.06, 0.0).finished());
  const sdv::ImagePyramid host(synthetic::renderHost(kW, kH), kLevels);
  const sdv::ImagePyramid target(synthetic::renderTarget(cam, T_t_h), kLevels);
  const sdv::ReferenceFrame ref = makeReference(cam, host);

  const Sophus::SE3d far = Sophus::SE3d::exp((Sophus::Vector6d() << 0, 0, 0.5, 0, -0.1, 0).finished());
  const Sophus::SE3d close = Sophus::SE3d::exp((Sophus::Vector6d() << 0.02, 0, -0.35, 0, 0.05, 0).finished());
  const sdv::FrameTracker tracker(sdv::TrackingSettings{});
  const auto res = tracker.track(ref, target, {far, close}, {});
  ASSERT_TRUE(res.ok);
  EXPECT_EQ(res.hypothesis, 1);
  EXPECT_LT((res.T_t_h * T_t_h.inverse()).log().norm(), 3e-3);
}

TEST(Tracker, MotionHypothesesStartWithConstantVelocity) {
  const Sophus::SE3d prevRef = Sophus::SE3d::exp((Sophus::Vector6d() << 0, 0, -1, 0, 0.01, 0).finished());
  const Sophus::SE3d vel = Sophus::SE3d::exp((Sophus::Vector6d() << 0, 0, -1, 0, 0.01, 0).finished());
  const auto h = sdv::makeMotionHypotheses(prevRef, vel);
  ASSERT_GE(h.size(), 4u);
  EXPECT_LT((h[0].log() - (vel * prevRef).log()).norm(), 1e-12);
  EXPECT_LT((h[3].log() - prevRef.log()).norm(), 1e-12);
}

INSTANTIATE_TEST_SUITE_P(Cameras, TrackerTest,
                         ::testing::Values(sdv::Camera::pinhole(250, 250, 159.5, 119.5, kW, kH),
                                           sdv::Camera::eucm(140, 140, 160.2, 119.7, 0.6, 1.1, kW, kH)),
                         [](const auto& info) { return info.param.isPinhole() ? "Pinhole" : "Fisheye"; });
