#include <cmath>
#include <random>

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

#include <sdv/loop_detector.h>

#include <synthetic_scene.h>

namespace {

constexpr int kW = 320, kH = 240;

sdv::Rig surroundRig() {
  const sdv::Camera cam = sdv::Camera::eucm(140, 140, 160.2, 119.7, 0.6, 1.1, kW, kH);
  return {{cam, cam, cam},
          {synthetic::room::cameraFromBody(0.0, {1.5, 0.0, 0.5}),
           synthetic::room::cameraFromBody(M_PI / 2, {0.5, 0.4, 0.5}),
           synthetic::room::cameraFromBody(M_PI, {-1.0, 0.0, 0.5})}};
}

// Keyframe record at body pose T_w_b with the true depth of every feature.
sdv::KeyframeRecord record(const sdv::Rig& rig, const Sophus::SE3d& T_w_b, int frameIndex) {
  sdv::KeyframeRecord r{frameIndex, T_w_b, {}};
  sdv::FeatureSettings settings;
  settings.featuresPerImage = 500;
  for (int c = 0; c < rig.size(); ++c) {
    const Sophus::SE3d T_c_w = rig.T_c_b[c] * T_w_b.inverse();
    cv::Mat img = synthetic::room::render(rig.cameras[c], T_c_w), gray;
    img.convertTo(gray, CV_8U);
    auto f = sdv::extractFeatures(rig.cameras[c], gray, settings);
    for (size_t i = 0; i < f.size(); ++i)
      f.rho[i] = static_cast<float>(synthetic::room::trueRho(T_c_w, f.bearings[i].cast<double>()));
    r.cameras.push_back(std::move(f));
  }
  return r;
}

}  // namespace

TEST(LoopDetector, VerifiesRevisitAcrossCameras) {
  const sdv::Rig rig = surroundRig();
  // Second visit a little to the side, driving the other way: the rear camera sees what the front camera saw.
  const Sophus::SE3d T_w_m(Sophus::SO3d::rotZ(0.0), Eigen::Vector3d(-2.0, 0.0, 0.0));
  const Sophus::SE3d T_w_q(Sophus::SO3d::rotZ(M_PI - 0.1), Eigen::Vector3d(-1.2, 0.4, 0.05));
  const auto m = record(rig, T_w_m, 0), q = record(rig, T_w_q, 500);
  sdv::LoopStats stats;
  const auto loop = sdv::verifyLoop(rig, q, m, sdv::LoopSettings{}, &stats);
  ASSERT_TRUE(loop.has_value()) << "few matches " << stats.fewMatches << ", few inliers " << stats.fewInliers;
  const Sophus::SE3d truth = T_w_q.inverse() * T_w_m;
  const Sophus::SE3d err = loop->T_q_m * truth.inverse();
  EXPECT_LT(err.translation().norm(), 0.03);
  EXPECT_LT(err.so3().log().norm(), 0.005);
  EXPECT_GT(loop->inliers, 60);
  EXPECT_LT(loop->rmsePixels, 1.5);
}

TEST(LoopDetector, RejectsGeometricallyInconsistentMatches) {
  const sdv::Rig rig = surroundRig();
  const Sophus::SE3d T_w_b(Sophus::SO3d::rotZ(0.3), Eigen::Vector3d(0.5, -0.5, 0.0));
  const auto m = record(rig, T_w_b, 0);
  // Same descriptors, scrambled geometry: matches exist but no single pose explains them.
  sdv::KeyframeRecord q = record(rig, T_w_b, 500);
  std::mt19937 rng(3);
  for (auto& f : q.cameras) {
    std::shuffle(f.bearings.begin(), f.bearings.end(), rng);
    std::shuffle(f.rho.begin(), f.rho.end(), rng);
  }
  sdv::LoopStats stats;
  EXPECT_FALSE(sdv::verifyLoop(rig, q, m, sdv::LoopSettings{}, &stats).has_value());
  EXPECT_EQ(stats.fewInliers + stats.fewMatches, 1);
}
