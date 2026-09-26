#include <cmath>

#include <gtest/gtest.h>

#include <sdv/immature_point.h>

#include <synthetic_scene.h>

namespace {

constexpr int kW = 320, kH = 240;

std::vector<sdv::ImmaturePoint> makePoints(const sdv::Camera& cam, const sdv::ImageLevel& host,
                                           const sdv::TraceSettings& settings) {
  std::vector<sdv::ImmaturePoint> pts;
  for (int v = 20; v < kH - 20; v += 9)
    for (int u = 20; u < kW - 20; u += 9)
      if (auto p = sdv::ImmaturePoint::create(cam, host, {double(u), double(v)}, settings);
          p && synthetic::trueRho(p->bearing()) > 0)
        pts.push_back(*p);
  return pts;
}

class ImmaturePointTest : public ::testing::TestWithParam<sdv::Camera> {};

}  // namespace

TEST_P(ImmaturePointTest, ConvergesToTrueDepth) {
  const sdv::Camera& cam = GetParam();
  const sdv::TraceSettings settings;
  const sdv::ImagePyramid host(synthetic::renderHost(kW, kH), 1);
  auto pts = makePoints(cam, host.level(0), settings);
  ASSERT_GT(pts.size(), 300u);

  const std::vector<Sophus::Vector6d> motions = {
      (Sophus::Vector6d() << -0.15, 0.02, -0.1, 0.0, 0.01, 0.0).finished(),
      (Sophus::Vector6d() << -0.3, 0.05, -0.2, 0.005, 0.015, -0.005).finished(),
      (Sophus::Vector6d() << -0.5, 0.08, -0.3, 0.01, 0.02, 0.0).finished()};
  for (const auto& m : motions) {
    sdv::HostTargetState state;
    state.T_t_h = Sophus::SE3d::exp(m);
    const sdv::ImagePyramid target(synthetic::renderTarget(cam, state.T_t_h), 1);
    for (auto& p : pts) p.trace(cam, target.level(0), state, settings);
  }

  int good = 0, accurate = 0, contained = 0;
  for (const auto& p : pts) {
    if (p.lastStatus() != sdv::TraceStatus::Good) continue;
    ++good;
    const double truth = synthetic::trueRho(p.bearing());
    if (std::abs(p.rho() - truth) < 0.03 * truth) ++accurate;
    if (truth >= p.rhoMin() && truth <= p.rhoMax()) ++contained;
  }
  EXPECT_GT(good, 0.6 * pts.size());
  EXPECT_GT(accurate, 0.95 * good);
  EXPECT_GT(contained, 0.9 * good);
}

TEST_P(ImmaturePointTest, RejectsUnrelatedImage) {
  const sdv::Camera& cam = GetParam();
  const sdv::TraceSettings settings;
  const sdv::ImagePyramid host(synthetic::renderHost(kW, kH), 1);
  auto pts = makePoints(cam, host.level(0), settings);

  cv::Mat noise(kH, kW, CV_32F);
  cv::randu(noise, 0.f, 255.f);
  const sdv::ImagePyramid target(noise, 1);
  sdv::HostTargetState state;
  state.T_t_h = Sophus::SE3d::exp((Sophus::Vector6d() << -0.3, 0, -0.1, 0, 0.01, 0).finished());
  int good = 0;
  for (auto& p : pts) good += p.trace(cam, target.level(0), state, settings) == sdv::TraceStatus::Good;
  EXPECT_LT(good, 0.1 * pts.size());
}

TEST(ImmaturePoint, SkipsWithoutParallax) {
  const auto cam = sdv::Camera::pinhole(250, 250, 159.5, 119.5, kW, kH);
  const sdv::TraceSettings settings;
  const sdv::ImagePyramid host(synthetic::renderHost(kW, kH), 1);
  auto p = sdv::ImmaturePoint::create(cam, host.level(0), {160, 120}, settings);
  ASSERT_TRUE(p);
  sdv::HostTargetState state;
  state.T_t_h = Sophus::SE3d::exp((Sophus::Vector6d() << 0, 0, 0, 0, 0.01, 0).finished());
  EXPECT_EQ(p->trace(cam, host.level(0), state, settings), sdv::TraceStatus::Skipped);
}

INSTANTIATE_TEST_SUITE_P(Cameras, ImmaturePointTest,
                         ::testing::Values(sdv::Camera::pinhole(250, 250, 159.5, 119.5, kW, kH),
                                           sdv::Camera::eucm(140, 140, 160.2, 119.7, 0.6, 1.1, kW, kH)),
                         [](const auto& info) { return info.param.isPinhole() ? "Pinhole" : "Fisheye"; });
