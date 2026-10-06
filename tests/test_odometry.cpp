#include <algorithm>
#include <optional>

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
  sdv::Odometry vo(cam, settings);

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

// A single camera mounted on a body: the body poses map back to the same camera trajectory.
TEST_P(OdometryTest, MountedCameraGivesSameCameraTrajectory) {
  const sdv::Camera& cam = GetParam();
  sdv::OdometrySettings settings;
  settings.levels = 4;
  settings.init.pointsLevel0 = 800;
  settings.candidatesPerKeyframe = 600;
  settings.targetActivePoints = 800;
  settings.kfFlow = 20.0;
  settings.kfTranslationFlow = 8.0;
  const Sophus::SE3d T_c_b = Sophus::SE3d::exp((Sophus::Vector6d() << 0.3, -1.2, 2.0, 1.2, -0.3, 0.8).finished());
  sdv::Odometry plain(cam, settings);
  sdv::Odometry mounted(sdv::Rig{{cam}, {T_c_b}}, settings);
  const Sophus::Vector6d v = (Sophus::Vector6d() << -0.06, 0.01, -0.03, 0.001, 0.004, 0.0).finished();
  for (int k = 0; k < 20; ++k) {
    cv::Mat img = synthetic::renderTarget(cam, Sophus::SE3d::exp(k * v));
    cv::GaussianBlur(img, img, cv::Size(0, 0), 1.0);
    plain.addFrame(img);
    mounted.addFrame(img);
  }
  const auto a = plain.poses(), b = mounted.poses();
  std::optional<Sophus::SE3d> firstA, firstB;
  int compared = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    ASSERT_EQ(a[i].has_value(), b[i].has_value()) << "frame " << i;
    if (!a[i]) continue;
    const Sophus::SE3d camA = *a[i], camB = *b[i] * T_c_b.inverse();  // T_w_c
    if (!firstA) firstA = camA, firstB = camB;
    const Sophus::SE3d relA = firstA->inverse() * camA, relB = firstB->inverse() * camB;
    EXPECT_LT((relA.translation() - relB.translation()).norm(), 1e-3 + 0.01 * relA.translation().norm()) << i;
    EXPECT_LT((relA.so3() * relB.so3().inverse()).log().norm(), 1e-3) << i;
    ++compared;
  }
  EXPECT_GT(compared, 10);
}

TEST_P(OdometryTest, StereoIsMetric) {
  const sdv::Camera& cam = GetParam();
  sdv::OdometrySettings settings;
  settings.levels = 4;
  settings.candidatesPerKeyframe = 600;
  settings.targetActivePoints = 800;
  settings.kfFlow = 20.0;
  settings.kfTranslationFlow = 8.0;
  const Sophus::SE3d T_r_l(Sophus::SO3d(), Eigen::Vector3d(-0.3, 0, 0));
  sdv::Odometry vo(sdv::Rig{{cam, cam}, {Sophus::SE3d(), T_r_l}}, settings);

  const Sophus::Vector6d v = (Sophus::Vector6d() << -0.06, 0.01, -0.03, 0.001, 0.004, 0.0).finished();
  std::vector<Sophus::SE3d> truth;  // T_w_c
  auto render = [&](const Sophus::SE3d& T_c_w) {
    cv::Mat img = synthetic::renderTarget(cam, T_c_w);
    cv::GaussianBlur(img, img, cv::Size(0, 0), 1.0);
    return img;
  };
  for (int k = 0; k < 20; ++k) {
    const Sophus::SE3d T_c_w = Sophus::SE3d::exp(k * v);
    truth.push_back(T_c_w.inverse());
    vo.addFrame(std::vector<cv::Mat>{render(T_c_w), render(T_r_l * T_c_w)});
  }
  const auto poses = vo.poses();
  ASSERT_TRUE(poses.front().has_value());  // stereo initialises on the first frame
  const double length = (truth.back().translation() - truth.front().translation()).norm();
  // No alignment at all: stereo fixes the scale and the first frame fixes the gauge. Single tracked frames can
  // be off by ~2 cm on a single plane (lateral shift vs. yaw), keyframes are refined by the window.
  double sumSq = 0;
  for (size_t i = 0; i < poses.size(); ++i) {
    ASSERT_TRUE(poses[i].has_value()) << "frame " << i;
    const double err = (poses[i]->translation() - truth[i].translation()).norm();
    EXPECT_LT(err, 0.025 * length) << "frame " << i;
    sumSq += err * err;
  }
  EXPECT_LT(std::sqrt(sumSq / poses.size()), 0.01 * length);
}

INSTANTIATE_TEST_SUITE_P(Cameras, OdometryTest,
                         ::testing::Values(sdv::Camera::pinhole(250, 250, 159.5, 119.5, kW, kH),
                                           sdv::Camera::eucm(140, 140, 160.2, 119.7, 0.6, 1.1, kW, kH)),
                         [](const auto& info) { return info.param.isPinhole() ? "Pinhole" : "Fisheye"; });

// Front, left and right fisheye with overlapping views in a closed room: metric trajectory without any alignment.
TEST(OdometryRig, SurroundFisheyeRigIsMetric) {
  const sdv::Camera cam = sdv::Camera::eucm(140, 140, 160.2, 119.7, 0.6, 1.1, kW, kH);
  const sdv::Rig rig{{cam, cam, cam},
                     {synthetic::room::cameraFromBody(0.0, {1.5, 0.0, 0.5}),
                      synthetic::room::cameraFromBody(M_PI / 2, {0.5, 0.4, 0.5}),
                      synthetic::room::cameraFromBody(-M_PI / 2, {0.5, -0.4, 0.5})}};
  sdv::OdometrySettings settings;
  settings.levels = 4;
  settings.candidatesPerKeyframe = 600;
  settings.targetActivePoints = 500;
  settings.kfFlow = 20.0;
  settings.kfTranslationFlow = 8.0;
  sdv::Odometry vo(rig, settings);

  std::vector<Sophus::SE3d> truth;  // T_w_b
  for (int k = 0; k < 30; ++k) {
    const Sophus::SE3d T_w_b = Sophus::SE3d::exp((Sophus::Vector6d() << 0.1 * k, 0, 0, 0, 0, 0.01 * k).finished());
    truth.push_back(T_w_b);
    std::vector<cv::Mat> images;
    for (int c = 0; c < rig.size(); ++c) {
      cv::Mat img = synthetic::room::render(cam, rig.T_c_b[c] * T_w_b.inverse());
      cv::GaussianBlur(img, img, cv::Size(0, 0), 1.0);
      images.push_back(img);
    }
    vo.addFrame(images);
  }
  const auto poses = vo.poses();
  ASSERT_TRUE(poses.front().has_value());  // several cameras initialise on the first frame
  const double length = (truth.back().translation() - truth.front().translation()).norm();
  double sumSq = 0;
  for (size_t i = 0; i < poses.size(); ++i) {
    ASSERT_TRUE(poses[i].has_value()) << "frame " << i;
    const double err = (poses[i]->translation() - truth[i].translation()).norm();
    EXPECT_LT(err, 0.02 * length) << "frame " << i;
    EXPECT_LT((poses[i]->so3() * truth[i].so3().inverse()).log().norm(), 2e-3) << "frame " << i;
    sumSq += err * err;
  }
  EXPECT_LT(std::sqrt(sumSq / poses.size()), 0.01 * length);
  std::vector<int> perCamera(rig.size(), 0);
  for (const auto& p : vo.mapPoints()) ++perCamera[p.camera];
  for (int c = 0; c < rig.size(); ++c) EXPECT_GT(perCamera[c], 100) << "camera " << c;
}

TEST(OdometryRig, KeyframesThatNoLongerSeeTheSceneLeaveTheWindow) {
  // A surround rig drives into a new area: from frame kSwitch on, every face but the rear wall looks different (as a
  // ramp tunnel seen from the level below). The old keyframes still see their rear-wall points with the rear camera,
  // so the visibility rule alone keeps them, but newer points are outliers in them.
  constexpr int kSwitch = 8, kFrames = 30;
  const sdv::Camera cam = sdv::Camera::eucm(140, 140, 160.2, 119.7, 0.6, 1.1, kW, kH);
  const sdv::Rig rig{{cam, cam, cam, cam},
                     {synthetic::room::cameraFromBody(0.0, {1.5, 0.0, 0.5}),
                      synthetic::room::cameraFromBody(M_PI / 2, {0.5, 0.4, 0.5}),
                      synthetic::room::cameraFromBody(-M_PI / 2, {0.5, -0.4, 0.5}),
                      synthetic::room::cameraFromBody(M_PI, {-0.5, 0.0, 0.5})}};
  auto render = [&](const Sophus::SE3d& T_c_w, bool switched) {
    const Sophus::SE3d T_w_c = T_c_w.inverse();
    cv::Mat img(cam.height, cam.width, CV_32F, cv::Scalar(0));
    for (int v = 0; v < cam.height; ++v)
      for (int u = 0; u < cam.width; ++u) {
        Eigen::Vector3d b;
        if (!cam.unproject(Eigen::Vector2d(u, v), b)) continue;
        const Eigen::Vector3d dir = T_w_c.so3() * b;
        double value = 0;
        const double s = synthetic::room::castRay(T_w_c.translation(), dir, &value);
        const Eigen::Vector3d p = T_w_c.translation() + s * dir;
        const bool rearWall = p.x() < synthetic::room::kMin.x() + 1e-6;
        if (switched && !rearWall) value = synthetic::room::texture(1.3 * p.z() + 3.0, 0.7 * (p.x() + p.y()) - 2.0);
        img.at<float>(v, u) = static_cast<float>(value);
      }
    cv::GaussianBlur(img, img, cv::Size(0, 0), 1.0);
    return img;
  };

  auto run = [&](double newerGoodFraction) {
    sdv::OdometrySettings settings;
    settings.levels = 4;
    settings.candidatesPerKeyframe = 600;
    settings.targetActivePoints = 500;
    settings.kfFlow = 20.0;
    settings.kfTranslationFlow = 8.0;
    settings.maxKeyframes = 7;
    settings.marginalizeNewerGoodFraction = newerGoodFraction;
    sdv::Odometry vo(rig, settings);
    for (int k = 0; k < kFrames; ++k) {
      const Sophus::SE3d T_w_b = Sophus::SE3d::trans(-4.0 + 0.1 * k, 0, 0);
      std::vector<cv::Mat> images;
      for (int c = 0; c < rig.size(); ++c) images.push_back(render(rig.T_c_b[c] * T_w_b.inverse(), k >= kSwitch));
      vo.addFrame(images);
    }
    return vo.windowKeyframeIndices();
  };
  auto oldest = [](const std::vector<int>& window) { return *std::ranges::min_element(window); };
  const auto before = run(0.0), after = run(0.3);
  EXPECT_LT(oldest(before), kSwitch);
  EXPECT_GE(oldest(after), kSwitch);
}
