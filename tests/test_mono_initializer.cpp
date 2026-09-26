#include <cmath>
#include <numbers>

#include <gtest/gtest.h>

#include <sdv/mono_initializer.h>

#include <synthetic_scene.h>

namespace {

constexpr int kW = 320, kH = 240, kLevels = 4;

class MonoInitializerTest : public ::testing::TestWithParam<sdv::Camera> {};

void checkInitialisation(const sdv::Camera& cam, const Sophus::Vector6d& velocity) {
  sdv::MonoInitSettings settings;
  settings.pointsLevel0 = 800;
  sdv::MonoInitializer init(cam, kLevels, settings);
  init.reset(sdv::ImagePyramid(synthetic::renderHost(kW, kH), kLevels));
  ASSERT_GT(init.numPoints(0), 400);

  sdv::MonoInitResult res;
  Sophus::SE3d T_true;
  for (int k = 1; k <= 10 && !res.initialized; ++k) {
    T_true = Sophus::SE3d::exp(k * velocity);
    res = init.addFrame(sdv::ImagePyramid(synthetic::renderTarget(cam, T_true, 1.0 + 0.02 * k, -2.0 * k), kLevels));
    ASSERT_FALSE(res.reset) << "frame " << k;
  }
  ASSERT_TRUE(res.initialized);
  EXPECT_GT(res.inlierRatio, 0.8);

  EXPECT_LT((res.T_t_h.so3() * T_true.so3().inverse()).log().norm(), 2e-3);
  const double cosAngle = res.T_t_h.translation().normalized().dot(T_true.translation().normalized());
  EXPECT_GT(cosAngle, std::cos(2.0 * std::numbers::pi / 180.0));

  const double toMetric = res.T_t_h.translation().norm() / T_true.translation().norm();
  int good = 0, accurate = 0;
  for (const auto& p : init.points()) {
    if (p.relativeSigma > 0.02) continue;
    ++good;
    const double truth = synthetic::trueRho(p.bearing);
    accurate += std::abs(p.rho * toMetric - truth) < 0.05 * truth;
  }
  EXPECT_GT(good, init.numPoints(0) / 2);
  EXPECT_GT(accurate, 0.9 * good);
}

}  // namespace

TEST_P(MonoInitializerTest, LateralMotion) {
  checkInitialisation(GetParam(), (Sophus::Vector6d() << -0.06, 0.01, -0.1, 0.002, 0.004, -0.001).finished());
}

TEST_P(MonoInitializerTest, ForwardMotionWithYaw) {
  checkInitialisation(GetParam(), (Sophus::Vector6d() << 0.01, 0.0, -0.25, 0.0, 0.02, 0.0).finished());
}

INSTANTIATE_TEST_SUITE_P(Cameras, MonoInitializerTest,
                         ::testing::Values(sdv::Camera::pinhole(250, 250, 159.5, 119.5, kW, kH),
                                           sdv::Camera::eucm(140, 140, 160.2, 119.7, 0.6, 1.1, kW, kH)),
                         [](const auto& info) { return info.param.isPinhole() ? "Pinhole" : "Fisheye"; });
