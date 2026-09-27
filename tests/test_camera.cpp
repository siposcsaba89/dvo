#include <gtest/gtest.h>

#include <sdv/camera.h>

namespace {

const sdv::Camera kFisheye = sdv::Camera::eucm(380.0, 381.0, 640.3, 400.7, 0.62, 1.05, 1280, 800);
const sdv::Camera kPinhole = sdv::Camera::pinhole(718.856, 718.856, 607.19, 185.22, 1232, 368);
// Rectified pinhole with a residual pincushion correction (negative alpha).
const sdv::Camera kPincushion = sdv::Camera::eucm(718.856, 718.856, 607.19, 185.22, -0.035, 1.0, 1232, 368);

void checkRoundTrip(const sdv::Camera& cam) {
  for (double v = 5; v < cam.height - 5; v += 37.3) {
    for (double u = 5; u < cam.width - 5; u += 41.7) {
      Eigen::Vector3d bearing;
      if (!cam.unproject(Eigen::Vector2d(u, v), bearing)) continue;
      EXPECT_NEAR(bearing.norm(), 1.0, 1e-12);
      for (double dist : {0.5, 3.0, 40.0}) {
        Eigen::Vector2d uv;
        ASSERT_TRUE(cam.project(Eigen::Vector3d(bearing * dist), uv));
        EXPECT_NEAR(uv.x(), u, 1e-8);
        EXPECT_NEAR(uv.y(), v, 1e-8);
      }
    }
  }
}

void checkJacobian(const sdv::Camera& cam, const Eigen::Vector3d& p) {
  Eigen::Vector2d uv;
  Eigen::Matrix<double, 2, 3> J;
  ASSERT_TRUE(cam.project(p, uv, J));
  constexpr double h = 1e-6;
  for (int k = 0; k < 3; ++k) {
    Eigen::Vector3d dp = Eigen::Vector3d::Zero();
    dp[k] = h;
    Eigen::Vector2d up, um;
    ASSERT_TRUE(cam.project(Eigen::Vector3d(p + dp), up));
    ASSERT_TRUE(cam.project(Eigen::Vector3d(p - dp), um));
    const Eigen::Vector2d num = (up - um) / (2 * h);
    EXPECT_NEAR(J(0, k), num.x(), 1e-5 * (1 + std::abs(num.x())));
    EXPECT_NEAR(J(1, k), num.y(), 1e-5 * (1 + std::abs(num.y())));
  }
}

}  // namespace

TEST(Camera, PinholeRoundTrip) { checkRoundTrip(kPinhole); }
TEST(Camera, FisheyeRoundTrip) { checkRoundTrip(kFisheye); }
TEST(Camera, PincushionRoundTrip) { checkRoundTrip(kPincushion); }

TEST(Camera, PinholeMatchesClassicFormula) {
  const Eigen::Vector3d p(1.2, -0.4, 7.5);
  Eigen::Vector2d uv;
  ASSERT_TRUE(kPinhole.project(p, uv));
  EXPECT_NEAR(uv.x(), kPinhole.fx * p.x() / p.z() + kPinhole.cx, 1e-10);
  EXPECT_NEAR(uv.y(), kPinhole.fy * p.y() / p.z() + kPinhole.cy, 1e-10);
  EXPECT_FALSE(kPinhole.project(Eigen::Vector3d(0.1, 0.1, -1.0), uv));
}

TEST(Camera, FisheyeSeesBeyond90Degrees) {
  Eigen::Vector2d uv;
  const double a = 95.0 * M_PI / 180.0;
  EXPECT_TRUE(kFisheye.project(Eigen::Vector3d(std::sin(a), 0, std::cos(a)), uv));
  EXPECT_FALSE(kFisheye.project(Eigen::Vector3d(0, 0, -1), uv));
}

TEST(Camera, ProjectionJacobians) {
  for (const auto& p : {Eigen::Vector3d(0.3, -0.2, 2.0), Eigen::Vector3d(-4.0, 1.5, 3.0),
                        Eigen::Vector3d(0.0, 0.0, 1.0)}) {
    checkJacobian(kPinhole, p);
    checkJacobian(kFisheye, p);
    checkJacobian(kPincushion, p);
  }
  checkJacobian(kFisheye, Eigen::Vector3d(2.0, 0.5, -0.1));
}

TEST(Camera, LevelScalingKeepsRayConsistent) {
  const Eigen::Vector3d p(0.7, -0.3, 1.1);
  Eigen::Vector2d uv0, uv2;
  ASSERT_TRUE(kFisheye.project(p, uv0));
  ASSERT_TRUE(kFisheye.atLevel(2).project(p, uv2));
  EXPECT_NEAR(uv2.x(), ((uv0.x() - 0.5) / 2 - 0.5) / 2, 1e-9);
  EXPECT_NEAR(uv2.y(), ((uv0.y() - 0.5) / 2 - 0.5) / 2, 1e-9);
}

TEST(Camera, NegativeAlphaIsPincushion) {
  // Off-axis rays land farther from the centre than with the plain pinhole.
  const Eigen::Vector3d p(0.8, 0.2, 1.0);
  Eigen::Vector2d pin, pc;
  ASSERT_TRUE(kPinhole.project(p, pin));
  ASSERT_TRUE(kPincushion.project(p, pc));
  EXPECT_GT(pc.x() - kPincushion.cx, pin.x() - kPinhole.cx);
}
