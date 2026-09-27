#include <cmath>
#include <random>

#include <gtest/gtest.h>

#include <sdv/global_ba.h>

#include <synthetic_scene.h>

namespace {

constexpr int kW = 320, kH = 240;

sdv::Rig surroundRig() {
  const sdv::Camera cam = sdv::Camera::eucm(140, 140, 160.2, 119.7, 0.6, 1.1, kW, kH);
  return {{cam, cam, cam},
          {synthetic::room::cameraFromBody(0.0, {1.5, 0.0, 0.5}),
           synthetic::room::cameraFromBody(M_PI / 2, {0.5, 0.4, 0.5}),
           synthetic::room::cameraFromBody(-M_PI / 2, {0.5, -0.4, 0.5})}};
}

sdv::KeyframeRecord record(const sdv::Rig& rig, const Sophus::SE3d& T_w_b, int frameIndex) {
  sdv::KeyframeRecord r{frameIndex, T_w_b, {}};
  sdv::FeatureSettings settings;
  settings.featuresPerImage = 400;
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

TEST(GlobalBA, RefinesPerturbedKeyframePoses) {
  const sdv::Rig rig = surroundRig();
  std::vector<Sophus::SE3d> truth;
  std::vector<sdv::KeyframeRecord> records;
  for (int k = 0; k < 8; ++k) {
    truth.push_back(Sophus::SE3d(Sophus::SO3d::rotZ(0.05 * k), Eigen::Vector3d(0.3 * k, 0.05 * k, 0)));
    records.push_back(record(rig, truth.back(), 10 * k));
  }
  std::mt19937 rng(5);
  std::normal_distribution<double> noise(0.0, 1.0);
  std::vector<Sophus::SE3d> initial = truth;
  for (size_t k = 1; k < initial.size(); ++k)
    initial[k] = initial[k] * Sophus::SE3d::exp((Sophus::Vector6d() << 0.03 * noise(rng), 0.03 * noise(rng),
                                                 0.03 * noise(rng), 0.004 * noise(rng), 0.004 * noise(rng),
                                                 0.004 * noise(rng)).finished());
  sdv::GlobalBASettings settings;
  settings.odometrySigmaFactor = 0;  // records carry the true poses; the test is about the features
  settings.gatePixels = 10.0;
  const auto result = sdv::bundleAdjustKeyframes(rig, records, initial, {}, settings);
  EXPECT_GT(result.observations, 1000u);
  EXPECT_LT(result.rmseAfter, 0.5 * result.rmseBefore);
  for (size_t k = 0; k < truth.size(); ++k) {
    const Sophus::SE3d err = result.T_w_b[k] * truth[k].inverse();
    EXPECT_LT(err.translation().norm(), 0.02) << "keyframe " << k;  // ~0.5 px at 5 m, ORB keypoint precision
    EXPECT_LT(err.so3().log().norm(), 0.002) << "keyframe " << k;
  }
}
