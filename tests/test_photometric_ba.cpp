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

namespace {

// Keyframes of the surround rig along a gentle curve: records (points with depth noise) and rendered images.
struct SyntheticRun {
  std::vector<Sophus::SE3d> truth;
  std::vector<sdv::KeyframeRecord> records;
  std::vector<std::vector<cv::Mat>> images;
};

SyntheticRun renderRun(const sdv::Rig& rig, std::mt19937& rng, int keyframes) {
  sdv::PointSelectorSettings selector;
  selector.targetPoints = 300;
  const sdv::PointSelector select(selector);
  std::normal_distribution<double> noise(0.0, 1.0);
  SyntheticRun run;
  for (int k = 0; k < keyframes; ++k) {
    run.truth.push_back(Sophus::SE3d(Sophus::SO3d::rotZ(0.04 * k), Eigen::Vector3d(0.25 * k, 0.03 * k, 0)));
    sdv::KeyframeRecord r{10 * k, run.truth.back(), std::vector<sdv::CameraFeatures>(rig.size()),
                          std::vector<sdv::AffineBrightness>(rig.size())};
    std::vector<cv::Mat> imgs;
    for (int c = 0; c < rig.size(); ++c) {
      const Sophus::SE3d T_c_w = rig.T_c_b[c] * run.truth.back().inverse();
      cv::Mat img = synthetic::room::render(rig.cameras[c], T_c_w);
      cv::GaussianBlur(img, img, cv::Size(0, 0), 1.0);
      const sdv::ImagePyramid pyr(img, 1);
      for (const auto& cand : select.select(pyr.level(0))) {
        Eigen::Vector3d b;
        const Eigen::Vector2d uv = cand.uv.cast<double>();
        if (!rig.cameras[c].unproject(uv, b)) continue;
        const double rho = synthetic::room::trueRho(T_c_w, b) * (1.0 + 0.03 * noise(rng));
        r.cameras[c].pointUv.push_back(uv.cast<float>());
        r.cameras[c].pointRho.push_back(static_cast<float>(rho));
      }
      imgs.push_back(img);
    }
    run.records.push_back(std::move(r));
    run.images.push_back(std::move(imgs));
  }
  return run;
}

}  // namespace

TEST(PhotometricBA, RefinesPerturbedExtrinsicsOnlyWhenAsked) {
  const sdv::Rig truthRig = surroundRig();
  std::mt19937 rng(5);
  const SyntheticRun run = renderRun(truthRig, rng, 8);
  // Left camera mounted 1 deg (yaw and pitch) and 3 cm off in the given rig; its records keep the true pixels.
  sdv::Rig rig = truthRig;
  rig.T_c_b[1] = Sophus::SE3d(Sophus::SO3d::rotY(0.0175) * Sophus::SO3d::rotX(0.0175), Eigen::Vector3d(0.03, 0, 0)) *
                 truthRig.T_c_b[1];
  sdv::PhotometricBASettings settings;
  settings.odometrySigmaFactor = 0;
  settings.maxInitialPixelError = 40.0;

  const auto fixed = sdv::photometricBundleAdjust(rig, run.records, run.images, run.truth, {}, settings);
  EXPECT_TRUE((fixed.T_c_b[1].matrix() - rig.T_c_b[1].matrix()).isZero(1e-12));

  settings.refineExtrinsics = true;
  const auto refined = sdv::photometricBundleAdjust(rig, run.records, run.images, run.truth, {}, settings);
  EXPECT_TRUE((refined.T_c_b[0].matrix() - rig.T_c_b[0].matrix()).isZero(1e-12));  // body reference
  for (int c = 1; c < rig.size(); ++c) {
    const Sophus::SE3d err = refined.T_c_b[c] * truthRig.T_c_b[c].inverse();
    EXPECT_LT(err.so3().log().norm(), 0.0035) << "camera " << c;  // 0.2 deg, from 1.4 deg
    EXPECT_LT(err.translation().norm(), 0.015) << "camera " << c;
  }
  EXPECT_LT(refined.rmseAfter, fixed.rmseAfter);
  // The per-pair check shows the left camera's static residuals improving.
  bool sawLeftStatic = false;
  for (const auto& p : refined.cameraPairs)
    if (p.sameKeyframe && (p.host == 1 || p.target == 1) && p.residuals > 20) {
      sawLeftStatic = true;
      EXPECT_LT(p.rmseAfter, p.rmseBefore) << p.host << " -> " << p.target;
    }
  EXPECT_TRUE(sawLeftStatic);
}

TEST(PhotometricBA, FixedRigScaleLetsTranslationsMove) {
  const sdv::Rig truthRig = surroundRig();
  std::mt19937 rng(5);
  const SyntheticRun run = renderRun(truthRig, rng, 8);
  // Left camera 4 cm off (with 1 deg), translations only weakly held: the sum of the camera distances stays.
  sdv::Rig rig = truthRig;
  rig.T_c_b[1] = Sophus::SE3d(Sophus::SO3d::rotY(0.0175), Eigen::Vector3d(0.04, 0, 0)) * truthRig.T_c_b[1];
  sdv::PhotometricBASettings settings;
  settings.odometrySigmaFactor = 0;
  settings.maxInitialPixelError = 40.0;
  settings.refineExtrinsics = true;
  settings.extrinsicSigmaT = 0.1;
  settings.extrinsicFixScale = true;
  const auto refined = sdv::photometricBundleAdjust(rig, run.records, run.images, run.truth, {}, settings);
  auto sumOfDistances = [](const std::vector<Sophus::SE3d>& T_c_b) {
    double s = 0;
    for (size_t i = 0; i < T_c_b.size(); ++i)
      for (size_t j = i + 1; j < T_c_b.size(); ++j)
        s += (T_c_b[i].inverse().translation() - T_c_b[j].inverse().translation()).norm();
    return s;
  };
  EXPECT_NEAR(sumOfDistances(refined.T_c_b), sumOfDistances(rig.T_c_b), 1e-4);
  const Sophus::SE3d err = refined.T_c_b[1] * truthRig.T_c_b[1].inverse();
  EXPECT_LT(err.so3().log().norm(), 0.0035);
  EXPECT_LT(err.translation().norm(), 0.025);  // from 0.04
}

TEST(PhotometricBA, RefinesPerturbedIntrinsicsWithFixedExtrinsics) {
  const sdv::Rig truthRig = surroundRig();
  std::mt19937 rng(7);
  const SyntheticRun run = renderRun(truthRig, rng, 8);
  // The left camera's focal lengths 2 % and principal point 2 px off in the given rig.
  sdv::Rig rig = truthRig;
  rig.cameras[1].fx *= 1.02, rig.cameras[1].fy *= 1.02, rig.cameras[1].cx += 2.0, rig.cameras[1].cy -= 2.0;
  sdv::PhotometricBASettings settings;
  settings.odometrySigmaFactor = 0;
  settings.maxInitialPixelError = 40.0;
  settings.refineIntrinsics = true;
  settings.intrinsicCameras = {1};
  settings.intrinsicSigmaFocal = 0.05;
  settings.intrinsicSigmaCenter = 5.0;
  const auto refined = sdv::photometricBundleAdjust(rig, run.records, run.images, run.truth, {}, settings);
  for (int c = 0; c < rig.size(); ++c)
    EXPECT_TRUE((refined.T_c_b[c].matrix() - rig.T_c_b[c].matrix()).isZero(1e-12)) << "camera " << c;
  EXPECT_EQ(refined.cameras[0].fx, rig.cameras[0].fx);
  const sdv::Camera& k = refined.cameras[1];
  const sdv::Camera& t = truthRig.cameras[1];
  EXPECT_LT(std::abs(k.fx / t.fx - 1), 0.005);
  EXPECT_LT(std::abs(k.fy / t.fy - 1), 0.005);
  EXPECT_LT(std::abs(k.cx - t.cx), 0.7);
  EXPECT_LT(std::abs(k.cy - t.cy), 0.7);
}

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

// Blocks of 4 keyframes (the rest fixed per block, borders shifted between sweeps) after a coarse joint solve on
// every 5th point reach the joint solution; without the coarse solve they drift ~0.5 mm per keyframe.
TEST(PhotometricBA, BlocksMatchTheJointAdjustment) {
  const sdv::Rig rig = surroundRig();
  std::mt19937 rng(7);
  const SyntheticRun run = renderRun(rig, rng, 12);
  std::normal_distribution<double> noise(0.0, 1.0);
  std::vector<Sophus::SE3d> initial = run.truth;
  for (size_t k = 1; k < initial.size(); ++k)
    initial[k] = initial[k] * Sophus::SE3d::exp((Sophus::Vector6d() << 0.01 * noise(rng), 0.01 * noise(rng),
                                                 0.01 * noise(rng), 0.002 * noise(rng), 0.002 * noise(rng),
                                                 0.002 * noise(rng)).finished());
  sdv::PhotometricBASettings settings;
  settings.odometrySigmaFactor = 0;
  settings.maxInitialPixelError = 40.0;
  const auto joint = sdv::photometricBundleAdjust(rig, run.records, run.images, initial, {}, settings);
  settings.blockKeyframes = 4;
  settings.blockCoarseResiduals = 30000;  // every 5th point in the coarse joint solve
  const auto blocks = sdv::photometricBundleAdjust(rig, run.records, run.images, initial, {}, settings);
  EXPECT_GT(blocks.residuals, 0.95 * joint.residuals);
  EXPECT_LT(blocks.rmseAfter, 1.02 * joint.rmseAfter);
  for (size_t k = 0; k < run.truth.size(); ++k) {
    const Sophus::SE3d err = blocks.T_w_b[k] * run.truth[k].inverse();
    EXPECT_LT(err.translation().norm(), 0.002) << "keyframe " << k;  // joint: < 0.6 mm, 0.1 mrad
    EXPECT_LT(err.so3().log().norm(), 0.0002) << "keyframe " << k;
  }
  ASSERT_EQ(blocks.pointDepthSigma.size(), joint.pointDepthSigma.size());
}

// Our Levenberg-Marquardt reaches Ceres' solution, jointly and in blocks (same residuals, robust loss, gauge).
TEST(PhotometricBA, CustomOptimizerMatchesCeres) {
  const sdv::Rig rig = surroundRig();
  std::mt19937 rng(7);
  const SyntheticRun run = renderRun(rig, rng, 12);
  std::normal_distribution<double> noise(0.0, 1.0);
  std::vector<Sophus::SE3d> initial = run.truth;
  for (size_t k = 1; k < initial.size(); ++k)
    initial[k] = initial[k] * Sophus::SE3d::exp((Sophus::Vector6d() << 0.01 * noise(rng), 0.01 * noise(rng),
                                                 0.01 * noise(rng), 0.002 * noise(rng), 0.002 * noise(rng),
                                                 0.002 * noise(rng)).finished());
  using Optimizer = sdv::PhotometricBASettings::Optimizer;
  struct Variant {
    int blockKeyframes, sparseBand, pcg;
  };
  for (const auto [blockKeyframes, sparseBand, pcg] : {Variant{0, -1, 0}, Variant{4, -1, 0}, Variant{0, 1, 0},
                                                       Variant{0, 1, 5}, Variant{4, 0, 5}}) {
    SCOPED_TRACE(testing::Message() << "blocks " << blockKeyframes << " band " << sparseBand << " pcg " << pcg);
    sdv::PhotometricBASettings settings;
    settings.maxInitialPixelError = 40.0;
    settings.blockKeyframes = blockKeyframes;
    settings.blockCoarseResiduals = 30000;
    settings.optimizer = Optimizer::Ceres;
    const auto ceres = sdv::photometricBundleAdjust(rig, run.records, run.images, initial, {}, settings);
    settings.optimizer = Optimizer::Custom;
    settings.sparseBand = sparseBand, settings.pcgIterations = pcg;
    const auto custom = sdv::photometricBundleAdjust(rig, run.records, run.images, initial, {}, settings);
    EXPECT_NEAR(custom.rmseAfter, ceres.rmseAfter, 0.01 * ceres.rmseAfter) << "blocks " << blockKeyframes;
    EXPECT_NEAR(static_cast<double>(custom.residuals), static_cast<double>(ceres.residuals), 0.01 * ceres.residuals);
    for (size_t k = 0; k < run.truth.size(); ++k) {
      const Sophus::SE3d d = custom.T_w_b[k] * ceres.T_w_b[k].inverse();
      EXPECT_LT(d.translation().norm(), 3e-4) << "blocks " << blockKeyframes << " keyframe " << k;
      EXPECT_LT(d.so3().log().norm(), 5e-5) << "blocks " << blockKeyframes << " keyframe " << k;
      const Sophus::SE3d err = custom.T_w_b[k] * run.truth[k].inverse();
      EXPECT_LT(err.translation().norm(), 0.002) << "keyframe " << k;
    }
    for (size_t k = 0; k < run.truth.size(); ++k)
      for (int c = 0; c < rig.size(); ++c) {
        EXPECT_NEAR(custom.affine[k][c].a, ceres.affine[k][c].a, 2e-3);
        EXPECT_NEAR(custom.affine[k][c].b, ceres.affine[k][c].b, 0.2);
      }
  }
}
