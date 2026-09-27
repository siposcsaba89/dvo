#include <cmath>
#include <random>

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

#include <sdv/photometric_ba.h>
#include <sdv/point_selector.h>

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

}  // namespace

TEST(PhotometricBA, RefinesPerturbedKeyframePosesAndDepths) {
  const sdv::Rig rig = surroundRig();
  sdv::PointSelectorSettings selector;
  selector.targetPoints = 300;
  const sdv::PointSelector select(selector);
  std::vector<Sophus::SE3d> truth;
  std::vector<sdv::KeyframeRecord> records;
  std::vector<std::vector<cv::Mat>> images;
  std::mt19937 rng(11);
  std::normal_distribution<double> noise(0.0, 1.0);
  for (int k = 0; k < 6; ++k) {
    truth.push_back(Sophus::SE3d(Sophus::SO3d::rotZ(0.04 * k), Eigen::Vector3d(0.25 * k, 0.03 * k, 0)));
    sdv::KeyframeRecord r{10 * k, truth.back(), std::vector<sdv::CameraFeatures>(rig.size()),
                          std::vector<sdv::AffineBrightness>(rig.size())};
    std::vector<cv::Mat> imgs;
    for (int c = 0; c < rig.size(); ++c) {
      const Sophus::SE3d T_c_w = rig.T_c_b[c] * truth.back().inverse();
      cv::Mat img = synthetic::room::render(rig.cameras[c], T_c_w);
      cv::GaussianBlur(img, img, cv::Size(0, 0), 1.0);
      const sdv::ImagePyramid pyr(img, 1);
      for (const auto& cand : select.select(pyr.level(0))) {
        Eigen::Vector3d b;
        const Eigen::Vector2d uv = cand.uv.cast<double>();
        if (!rig.cameras[c].unproject(uv, b)) continue;
        // Depth with 3 % noise: the adjustment has to fix it.
        const double rho = synthetic::room::trueRho(T_c_w, b) * (1.0 + 0.03 * noise(rng));
        r.cameras[c].pointUv.push_back(uv.cast<float>());
        r.cameras[c].pointRho.push_back(static_cast<float>(rho));
      }
      imgs.push_back(img);
    }
    records.push_back(std::move(r));
    images.push_back(std::move(imgs));
  }
  std::vector<Sophus::SE3d> initial = truth;
  for (size_t k = 1; k < initial.size(); ++k)
    initial[k] = initial[k] * Sophus::SE3d::exp((Sophus::Vector6d() << 0.01 * noise(rng), 0.01 * noise(rng),
                                                 0.01 * noise(rng), 0.002 * noise(rng), 0.002 * noise(rng),
                                                 0.002 * noise(rng)).finished());
  sdv::PhotometricBASettings settings;
  settings.odometrySigmaFactor = 0;
  settings.maxInitialPixelError = 40.0;
  const auto result = sdv::photometricBundleAdjust(rig, records, images, initial, {}, settings);
  EXPECT_GT(result.residuals, 2000u);
  EXPECT_LT(result.rmseAfter, 0.3 * result.rmseBefore);
  for (size_t k = 0; k < truth.size(); ++k) {
    const Sophus::SE3d err = result.T_w_b[k] * truth[k].inverse();
    EXPECT_LT(err.translation().norm(), 0.003) << "keyframe " << k;
    EXPECT_LT(err.so3().log().norm(), 0.0007) << "keyframe " << k;
  }
}
