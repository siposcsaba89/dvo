#include <cmath>

#include <gtest/gtest.h>

#include <sdv/photometric.h>

namespace {

constexpr int kW = 320, kH = 240;

cv::Mat makeSmoothImage(double gain = 1.0, double offset = 0.0) {
  cv::Mat img(kH, kW, CV_32F);
  for (int v = 0; v < kH; ++v)
    for (int u = 0; u < kW; ++u) {
      const double i = 128 + 80 * std::sin(u / 25.0) * std::cos(v / 20.0) + 40 * std::sin((u + v) / 40.0);
      img.at<float>(v, u) = static_cast<float>(gain * i + offset);
    }
  return img;
}

const sdv::Camera kPinhole = sdv::Camera::pinhole(250, 250, 159.5, 119.5, kW, kH);
const sdv::Camera kFisheye = sdv::Camera::eucm(140, 140, 160.2, 119.7, 0.6, 1.1, kW, kH);

const Sophus::SE3d kTth = Sophus::SE3d::exp((Sophus::Vector6d() << -0.1, 0.05, -0.4, 0.02, -0.03, 0.01).finished());

Eigen::Vector3d bearingAt(const sdv::Camera& cam, double u, double v) {
  Eigen::Vector3d b;
  EXPECT_TRUE(cam.unproject(Eigen::Vector2d(u, v), b));
  return b;
}

Eigen::Vector2d project(const Eigen::Vector3d& b, double rho, const Sophus::SE3d& T, const sdv::Camera& cam) {
  Eigen::Vector2d uv;
  EXPECT_TRUE(sdv::projectBearing(b, rho, T, cam, uv));
  return uv;
}

class PhotometricTest : public ::testing::TestWithParam<sdv::Camera> {};

}  // namespace

TEST_P(PhotometricTest, ProjectionJacobians) {
  const sdv::Camera& cam = GetParam();
  constexpr double h = 1e-6;
  for (double rho : {0.0, 0.2, 1.5}) {
    const Eigen::Vector3d b = bearingAt(cam, 140.3, 101.8);
    Eigen::Vector2d uv, dRho;
    Eigen::Matrix<double, 2, 6> dPose;
    ASSERT_TRUE(sdv::projectBearing(b, rho, kTth, cam, uv, &dPose, &dRho));
    for (int k = 0; k < 6; ++k) {
      const Sophus::Vector6d d = h * Sophus::Vector6d::Unit(k);
      const Eigen::Vector2d num =
          (project(b, rho, Sophus::SE3d::exp(d) * kTth, cam) - project(b, rho, Sophus::SE3d::exp(-d) * kTth, cam)) /
          (2 * h);
      EXPECT_LT((dPose.col(k) - num).norm(), 1e-5 * (1 + num.norm())) << "rho " << rho << " k " << k;
    }
    const Eigen::Vector2d numRho = (project(b, rho + h, kTth, cam) - project(b, rho - h, kTth, cam)) / (2 * h);
    EXPECT_LT((dRho - numRho).norm(), 1e-5 * (1 + numRho.norm()));
  }
}

TEST_P(PhotometricTest, AbsolutePoseChain) {
  const sdv::Camera& cam = GetParam();
  const Sophus::SE3d T_h_w = Sophus::SE3d::exp((Sophus::Vector6d() << 1, 2, 3, 0.1, 0.2, -0.3).finished());
  const Sophus::SE3d T_t_w = kTth * T_h_w;
  const Eigen::Vector3d b = bearingAt(cam, 170.0, 130.0);
  const double rho = 0.3;

  Eigen::Vector2d uv;
  Eigen::Matrix<double, 2, 6> dRel;
  ASSERT_TRUE(sdv::projectBearing(b, rho, T_t_w * T_h_w.inverse(), cam, uv, &dRel));
  const Eigen::Matrix<double, 2, 6> dHost = dRel * sdv::dRelativeDHost(T_t_w * T_h_w.inverse());
  const Eigen::Matrix<double, 2, 6> dTarget = dRel * sdv::dRelativeDTarget();

  constexpr double h = 1e-6;
  for (int k = 0; k < 6; ++k) {
    const Sophus::SE3d ep = Sophus::SE3d::exp(h * Sophus::Vector6d::Unit(k));
    const Sophus::SE3d em = Sophus::SE3d::exp(-h * Sophus::Vector6d::Unit(k));
    const Eigen::Vector2d numHost = (project(b, rho, T_t_w * (ep * T_h_w).inverse(), cam) -
                                     project(b, rho, T_t_w * (em * T_h_w).inverse(), cam)) / (2 * h);
    const Eigen::Vector2d numTarget = (project(b, rho, ep * T_t_w * T_h_w.inverse(), cam) -
                                       project(b, rho, em * T_t_w * T_h_w.inverse(), cam)) / (2 * h);
    EXPECT_LT((dHost.col(k) - numHost).norm(), 1e-5 * (1 + numHost.norm()));
    EXPECT_LT((dTarget.col(k) - numTarget).norm(), 1e-5 * (1 + numTarget.norm()));
  }
}

TEST_P(PhotometricTest, ZeroResidualUnderBrightnessChange) {
  const sdv::Camera& cam = GetParam();
  const sdv::ImagePyramid host(makeSmoothImage(), 1);
  const double da = 0.2, bh = 10.0, bt = 5.0;
  const sdv::ImagePyramid target(makeSmoothImage(std::exp(da), bt - std::exp(da) * bh), 1);
  const sdv::PhotometricSettings settings;

  const auto p = sdv::makePatternPoint(cam, host.level(0), {150, 110}, settings);
  ASSERT_TRUE(p);
  sdv::HostTargetState state;
  state.host = {0.1, bh};
  state.target = {0.1 + da, bt};
  sdv::PatternResidual res;
  ASSERT_TRUE(sdv::evaluatePatternResidual(*p, 0.5, state, cam, target.level(0), settings, res));
  for (const auto& px : res.pixels) EXPECT_NEAR(px.r, 0.0, 1e-3);
  EXPECT_NEAR(res.energy, 0.0, 1e-5);
}

TEST_P(PhotometricTest, ResidualJacobians) {
  const sdv::Camera& cam = GetParam();
  const sdv::ImagePyramid host(makeSmoothImage(), 1);
  const sdv::ImagePyramid target(makeSmoothImage(1.1, -7.0), 1);
  const sdv::PhotometricSettings settings;
  const auto p = sdv::makePatternPoint(cam, host.level(0), {150, 110}, settings);
  ASSERT_TRUE(p);

  sdv::HostTargetState state;
  state.T_t_h = kTth;
  state.host = {0.05, 3.0};
  state.target = {-0.1, 8.0};
  state.exposureRatio = 1.3;
  const double rho = 0.25;

  sdv::PatternResidual res;
  ASSERT_TRUE(sdv::evaluatePatternResidual(*p, rho, state, cam, target.level(0), settings, res));

  auto residuals = [&](const sdv::HostTargetState& s, double r) {
    sdv::PatternResidual out;
    EXPECT_TRUE(sdv::evaluatePatternResidual(*p, r, s, cam, target.level(0), settings, out));
    Eigen::Matrix<double, sdv::kPatternSize, 1> v;
    for (int k = 0; k < sdv::kPatternSize; ++k) v[k] = out.pixels[k].r;
    return v;
  };

  Eigen::Matrix<double, sdv::kPatternSize, 6> aPose;
  Eigen::Matrix<double, sdv::kPatternSize, 1> aRho;
  Eigen::Matrix<double, sdv::kPatternSize, 4> aAff;
  for (int k = 0; k < sdv::kPatternSize; ++k) {
    aPose.row(k) = res.pixels[k].dPose;
    aRho[k] = res.pixels[k].dRho;
    aAff.row(k) = res.pixels[k].dAffine;
  }

  // Image gradients are central differences while bilinear interpolation is piecewise linear,
  // so geometric derivatives agree only approximately; affine derivatives are exact.
  constexpr double hp = 1e-4;
  for (int k = 0; k < 6; ++k) {
    auto sp = state, sm = state;
    sp.T_t_h = Sophus::SE3d::exp(hp * Sophus::Vector6d::Unit(k)) * state.T_t_h;
    sm.T_t_h = Sophus::SE3d::exp(-hp * Sophus::Vector6d::Unit(k)) * state.T_t_h;
    const auto num = ((residuals(sp, rho) - residuals(sm, rho)) / (2 * hp)).eval();
    EXPECT_LT((aPose.col(k) - num).norm(), 0.05 * num.norm() + 0.05) << "pose " << k;
  }
  const auto numRho = ((residuals(state, rho + hp) - residuals(state, rho - hp)) / (2 * hp)).eval();
  EXPECT_LT((aRho - numRho).norm(), 0.05 * numRho.norm() + 0.05);

  constexpr double ha = 1e-6;
  for (int k = 0; k < 4; ++k) {
    auto sp = state, sm = state;
    double* pp[4] = {&sp.host.a, &sp.host.b, &sp.target.a, &sp.target.b};
    double* pm[4] = {&sm.host.a, &sm.host.b, &sm.target.a, &sm.target.b};
    *pp[k] += ha;
    *pm[k] -= ha;
    const auto num = ((residuals(sp, rho) - residuals(sm, rho)) / (2 * ha)).eval();
    EXPECT_LT((aAff.col(k) - num).norm(), 1e-5 * (1 + num.norm())) << "affine " << k;
  }
}

TEST(Photometric, HuberWeightMatchesEnergyDerivative) {
  constexpr double k = 9.0, h = 1e-6;
  for (double r : {-30.0, -9.0, -2.0, 0.5, 8.9, 25.0}) {
    const double dE = (sdv::huberEnergy(r + h, k) - sdv::huberEnergy(r - h, k)) / (2 * h);
    EXPECT_NEAR(sdv::huberWeight(r, k) * r, dE, 1e-5);
  }
}

INSTANTIATE_TEST_SUITE_P(Cameras, PhotometricTest, ::testing::Values(kPinhole, kFisheye),
                         [](const auto& info) { return info.param.isPinhole() ? "Pinhole" : "Fisheye"; });
