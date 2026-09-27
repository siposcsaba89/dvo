#include <cmath>
#include <filesystem>

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

#include <sdv/keyframe_features.h>

#include <synthetic_scene.h>

namespace {

cv::Mat roomImage(const sdv::Camera& cam, const Sophus::SE3d& T_c_w) {
  cv::Mat img = synthetic::room::render(cam, T_c_w), out;
  img.convertTo(out, CV_8U);
  return out;
}

}  // namespace

TEST(KeyframeFeatures, ExtractsSpreadKeypointsWithBearings) {
  const sdv::Camera cam = sdv::Camera::eucm(140, 140, 160.2, 119.7, 0.6, 1.1, 320, 240);
  const Sophus::SE3d T_c_w = synthetic::room::cameraFromBody(0.3, {0.5, 0.2, 0.3});
  sdv::FeatureSettings settings;
  settings.featuresPerImage = 400;
  const auto f = sdv::extractFeatures(cam, roomImage(cam, T_c_w), settings);
  ASSERT_GT(f.size(), 200u);
  EXPECT_EQ(f.descriptors.rows, static_cast<int>(f.size()));
  EXPECT_EQ(f.descriptors.cols, 32);
  int left = 0;
  for (size_t i = 0; i < f.size(); ++i) {
    EXPECT_NEAR(f.bearings[i].norm(), 1.0f, 1e-5f);
    Eigen::Vector2d uv;
    ASSERT_TRUE(cam.project(f.bearings[i].cast<double>().eval(), uv));
    EXPECT_NEAR(uv.x(), f.keypoints[i].pt.x, 1e-2);
    left += f.keypoints[i].pt.x < 160;
  }
  EXPECT_GT(left, static_cast<int>(f.size()) / 4);
  EXPECT_LT(left, static_cast<int>(f.size()) * 3 / 4);
}

TEST(KeyframeFeatures, DepthOnlyFromConsistentNeighbours) {
  sdv::CameraFeatures f;
  for (float x : {10.f, 50.f, 90.f}) {
    f.keypoints.emplace_back(cv::Point2f(x, 20.f), 31.f);
    f.bearings.emplace_back(0.f, 0.f, 1.f);
    f.rho.push_back(0.f);
  }
  const std::vector<Eigen::Vector2f> uv = {{11, 20}, {9, 21}, {51, 20}, {49, 20}, {200, 200}};
  const std::vector<float> rho = {0.50f, 0.51f, 0.2f, 0.4f, 1.0f};
  sdv::assignDepth(f, uv, rho, sdv::FeatureSettings{});
  EXPECT_NEAR(f.rho[0], 0.505f, 0.006f);  // two agreeing points
  EXPECT_EQ(f.rho[1], 0.f);  // depth edge
  EXPECT_EQ(f.rho[2], 0.f);  // nothing near
  EXPECT_EQ(f.numWithDepth(), 1);
}

TEST(KeyframeFeatures, RecordsRoundTrip) {
  const sdv::Camera cam = sdv::Camera::eucm(140, 141, 160.2, 119.7, 0.6, 1.1, 320, 240);
  const sdv::Rig rig{{cam, cam}, {Sophus::SE3d(), synthetic::room::cameraFromBody(M_PI, {-1, 0, 0.5})}};
  sdv::KeyframeRecord r{42, Sophus::SE3d::exp((Sophus::Vector6d() << 1, 2, 3, 0.1, 0.2, 0.3).finished()), {}};
  for (int c = 0; c < 2; ++c) {
    auto f = sdv::extractFeatures(cam, roomImage(cam, rig.T_c_b[c]), sdv::FeatureSettings{});
    if (!f.rho.empty()) f.rho[0] = 0.25f;
    r.cameras.push_back(f);
  }
  const auto file = std::filesystem::temp_directory_path() / "sdv_records_test.bin";
  sdv::saveKeyframeRecords(file, rig, {r});
  sdv::Rig loadedRig;
  const auto loaded = sdv::loadKeyframeRecords(file, &loadedRig);
  std::filesystem::remove(file);
  ASSERT_EQ(loaded.size(), 1u);
  EXPECT_EQ(loaded[0].frameIndex, 42);
  EXPECT_LT((loaded[0].T_w_b.log() - r.T_w_b.log()).norm(), 1e-9);
  ASSERT_EQ(loadedRig.size(), 2);
  EXPECT_EQ(loadedRig.cameras[1].fy, 141);
  EXPECT_LT((loadedRig.T_c_b[1].log() - rig.T_c_b[1].log()).norm(), 1e-9);
  for (int c = 0; c < 2; ++c) {
    const auto& a = r.cameras[c];
    const auto& b = loaded[0].cameras[c];
    ASSERT_EQ(a.size(), b.size());
    EXPECT_EQ(cv::norm(a.descriptors, b.descriptors, cv::NORM_HAMMING), 0.0);
    for (size_t i = 0; i < a.size(); ++i) {
      EXPECT_EQ(a.keypoints[i].pt, b.keypoints[i].pt);
      EXPECT_EQ(a.keypoints[i].octave, b.keypoints[i].octave);
      EXPECT_EQ(a.rho[i], b.rho[i]);
    }
  }
}
