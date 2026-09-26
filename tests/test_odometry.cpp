#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

#include <sdv/eval/trajectory.h>
#include <sdv/odometry.h>

#include <synthetic_scene.h>

namespace {

constexpr int kW = 320, kH = 240;

class OdometryTest : public ::testing::TestWithParam<sdv::Camera> {};

}  // namespace

TEST_P(OdometryTest, TracksSyntheticSequence) {
  const sdv::Camera& cam = GetParam();
  sdv::OdometrySettings settings;
  settings.levels = 4;
  settings.init.pointsLevel0 = 800;
  settings.candidatesPerKeyframe = 600;
  settings.targetActivePoints = 800;
  settings.kfFlow = 20.0;
  settings.kfTranslationFlow = 8.0;
  sdv::MonoOdometry vo(cam, settings);

  const Sophus::Vector6d v = (Sophus::Vector6d() << -0.06, 0.01, -0.03, 0.001, 0.004, 0.0).finished();
  std::vector<Sophus::SE3d> truth;  // T_w_c
  int keyframes = 0;
  for (int k = 0; k < 30; ++k) {
    const Sophus::SE3d T_c_w = Sophus::SE3d::exp(k * v);
    truth.push_back(T_c_w.inverse());
    cv::Mat img = synthetic::renderTarget(cam, T_c_w, 1.0 + 0.005 * k, -0.5 * k);
    cv::GaussianBlur(img, img, cv::Size(0, 0), 1.0);
    keyframes += vo.addFrame(img).keyframe;
  }
  ASSERT_TRUE(vo.initialized());
  EXPECT_GT(keyframes, 3);

  const auto poses = vo.poses();
  std::vector<Sophus::SE3d> est, gt;
  for (size_t i = 0; i < poses.size(); ++i)
    if (poses[i]) {
      est.push_back(*poses[i]);
      gt.push_back(truth[i]);
    }
  EXPECT_GE(est.size(), 20u);  // frames before the initialisation host have no pose
  const auto ate = sdv::absoluteTrajectoryError(gt, est, true);
  const double length = (gt.back().translation() - gt.front().translation()).norm();
  EXPECT_LT(ate.rmse, 0.01 * length);
  // Relative rotations: a straight trajectory leaves the alignment's rotation about its axis undetermined.
  for (size_t i = 1; i < est.size(); ++i) {
    const Sophus::SO3d relEst = est[0].so3().inverse() * est[i].so3();
    const Sophus::SO3d relGt = gt[0].so3().inverse() * gt[i].so3();
    EXPECT_LT((relEst * relGt.inverse()).log().norm(), 2e-3) << "frame " << i;
  }
  EXPECT_GT(vo.mapPoints().size(), 500u);
}

INSTANTIATE_TEST_SUITE_P(Cameras, OdometryTest,
                         ::testing::Values(sdv::Camera::pinhole(250, 250, 159.5, 119.5, kW, kH),
                                           sdv::Camera::eucm(140, 140, 160.2, 119.7, 0.6, 1.1, kW, kH)),
                         [](const auto& info) { return info.param.isPinhole() ? "Pinhole" : "Fisheye"; });
