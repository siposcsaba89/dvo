#include <filesystem>
#include <fstream>
#include <memory>

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <sdv/camera.h>
#include <sdv/io/camera_config.h>
#include <sdv/io/frame_source.h>
#include <sdv/io/rig_config.h>
#include <sdv/validity_mask.h>

namespace {

std::filesystem::path tempDir(const char* name) {
  const auto dir = std::filesystem::temp_directory_path() / "sdv_tests" / name;
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  return dir;
}

cv::Mat circleMask(int w, int h) {
  cv::Mat m(h, w, CV_8UC1, cv::Scalar(0));
  cv::circle(m, {w / 2, h / 2}, h / 2 - 4, cv::Scalar(255), cv::FILLED);
  return m;
}

}  // namespace

TEST(ValidityMask, ErodesAndPoolsLevels) {
  cv::Mat img(64, 64, CV_8UC1, cv::Scalar(255));
  img(cv::Rect(0, 0, 20, 64)).setTo(100);  // columns 0-19 invalid (<= 128)
  const sdv::ValidityMask mask(img, 2);
  EXPECT_FALSE(mask.valid(0, 19, 30));
  EXPECT_FALSE(mask.valid(0, 21, 30));  // eroded
  EXPECT_TRUE(mask.valid(0, 22, 30));
  EXPECT_FALSE(mask.valid(0, 62, 30));  // image border counts as invalid
  EXPECT_EQ(mask.width(1), 32);
  // Level 1 pixel 10 covers input columns 20, 21 (valid); pixel 11 is valid after erosion only from 12 on.
  EXPECT_FALSE(mask.valid(1, 11, 15));
  EXPECT_TRUE(mask.valid(1, 12, 15));
}

TEST(ValidityMask, CameraIsInsideUsesMaskAtItsLevel) {
  sdv::Camera cam = sdv::Camera::pinhole(100, 100, 63.5, 63.5, 128, 128);
  cam.mask = std::make_shared<const sdv::ValidityMask>(circleMask(128, 128));
  EXPECT_TRUE(cam.isInside(64, 64, 2));
  EXPECT_FALSE(cam.isInside(5, 5, 0));  // inside the rectangle, outside the circle
  const sdv::Camera c2 = cam.atLevel(2);
  EXPECT_EQ(c2.maskLevel, 2);
  EXPECT_TRUE(c2.isInside(16, 16, 1));
  EXPECT_FALSE(c2.isInside(1.5, 1.5, 0));
  EXPECT_EQ(c2.maskImage().cols, 32);
}

TEST(CameraConfig, LoadsYamlAndMask) {
  const auto dir = tempDir("camera_config");
  cv::imwrite((dir / "mask.png").string(), circleMask(200, 100));
  std::ofstream(dir / "cam.yaml") << "width: 200\nheight: 100\nfx: 80.5\nfy: 81\ncx: 99.5\ncy: 49.5\n"
                                     "alpha: 0.6\nbeta: 1.1\nmask: mask.png\n";
  const sdv::CameraConfig config = sdv::loadCameraConfig(dir / "cam.yaml");
  EXPECT_EQ(config.camera.width, 200);
  EXPECT_DOUBLE_EQ(config.camera.fy, 81.0);
  EXPECT_DOUBLE_EQ(config.camera.alpha, 0.6);
  EXPECT_DOUBLE_EQ(config.camera.beta, 1.1);
  ASSERT_FALSE(config.mask.empty());
  EXPECT_EQ(config.mask.cols, 200);

  std::ofstream(dir / "pinhole.yaml") << "width: 200\nheight: 100\nfx: 80\nfy: 80\ncx: 99.5\ncy: 49.5\n";
  const sdv::CameraConfig pinhole = sdv::loadCameraConfig(dir / "pinhole.yaml");
  EXPECT_TRUE(pinhole.camera.isPinhole());
  EXPECT_TRUE(pinhole.mask.empty());

  std::ofstream(dir / "bad.yaml") << "width: 200\nheight: 100\nfx: 80\n";
  EXPECT_THROW(sdv::loadCameraConfig(dir / "bad.yaml"), std::runtime_error);
}

TEST(RigConfig, LoadsCamerasAndExtrinsics) {
  const auto dir = tempDir("rig_config");
  std::ofstream(dir / "front.yaml") << "width: 200\nheight: 100\nfx: 80\nfy: 80\ncx: 99.5\ncy: 49.5\nalpha: 0.5\n";
  // Front camera looking along body x (camera z), rear camera turned by 180 deg about body z.
  std::ofstream(dir / "rig.yaml")
      << "cameras:\n"
         "  - name: front\n"
         "    camera: front.yaml\n"
         "    video: front.h264\n"
         "    T_body_camera:\n"
         "      translation: [2.0, 0.1, 1.5]\n"
         "      rotation_matrix: [0, 0, 1, -1, 0, 0, 0, -1, 0]\n"
         "  - name: rear\n"
         "    camera: {width: 200, height: 100, fx: 70, fy: 70, cx: 99.5, cy: 49.5}\n"
         "    video: " + std::filesystem::absolute("/abs/rear.h264").generic_string() + "\n"
         "    frame_offset: 3\n"
         "    T_body_camera:\n"
         "      translation: [-1.0, 0.0, 1.2]\n"
         "      rotation_quaternion_wxyz: [0.5, -0.5, -0.5, 0.5]\n";
  const sdv::RigConfig rig = sdv::loadRigConfig(dir / "rig.yaml");
  ASSERT_EQ(rig.cameras.size(), 2u);
  const auto& front = rig.cameras[0];
  EXPECT_EQ(front.name, "front");
  EXPECT_DOUBLE_EQ(front.camera.camera.alpha, 0.5);
  EXPECT_EQ(front.video, dir / "front.h264");
  EXPECT_EQ(front.frameOffset, 0);
  EXPECT_TRUE((front.T_b_c * Eigen::Vector3d(0, 0, 1) - Eigen::Vector3d(3.0, 0.1, 1.5)).norm() < 1e-9);
  const auto& rear = rig.cameras[1];
  EXPECT_DOUBLE_EQ(rear.camera.camera.fx, 70.0);
  EXPECT_EQ(rear.frameOffset, 3);
  EXPECT_EQ(rear.video, std::filesystem::absolute("/abs/rear.h264"));
  // Camera z maps to body -x; camera x (image right) to body +y (left of the vehicle, seen from behind).
  EXPECT_TRUE((rear.T_b_c.so3() * Eigen::Vector3d(0, 0, 1) - Eigen::Vector3d(-1, 0, 0)).norm() < 1e-9);
  EXPECT_TRUE((rear.T_b_c.so3() * Eigen::Vector3d(1, 0, 0) - Eigen::Vector3d(0, 1, 0)).norm() < 1e-9);

  std::ofstream(dir / "bad.yaml") << "cameras:\n  - camera: front.yaml\n    T_body_camera:\n      translation: [0, 0, 0]\n"
                                     "      rotation_matrix: [1, 0, 0, 0, 1, 0, 0, 0, 2]\n";
  EXPECT_THROW(sdv::loadRigConfig(dir / "bad.yaml"), std::runtime_error);
}

TEST(RigConfig, LoadsAimRecord) {
  const auto dir = tempDir("rig_config_aim");
  std::ofstream(dir / "front.yaml") << "width: 200\nheight: 100\nfx: 80\nfy: 80\ncx: 99.5\ncy: 49.5\n";
  const std::string pose = "    T_body_camera:\n      translation: [2.0, 0.1, 1.5]\n"
                           "      rotation_matrix: [0, 0, 1, -1, 0, 0, 0, -1, 0]\n";
  std::ofstream(dir / "rig.yaml") << "aim_record:\n  record: record\n  calibration: /abs/config.zip\n"
                                     "cameras:\n  - name: F_FISHEYE_C\n    camera: front.yaml\n" + pose;
  const sdv::RigConfig rig = sdv::loadRigConfig(dir / "rig.yaml");
  EXPECT_EQ(rig.aimRecord, dir / "record");
  EXPECT_EQ(rig.aimCalibration, std::filesystem::path("/abs/config.zip"));
  ASSERT_EQ(rig.cameras.size(), 1u);
  EXPECT_TRUE(rig.cameras[0].video.empty());
  EXPECT_THROW(sdv::openRigStreams(rig, {"B_FISHEYE_C"}), std::invalid_argument);

  // Per-camera videos and offsets do not mix with a recording.
  std::ofstream(dir / "mixed.yaml") << "aim_record:\n  record: record\n  calibration: config.zip\n"
                                       "cameras:\n  - name: F_FISHEYE_C\n    camera: front.yaml\n    frame_offset: 1\n" +
                                           pose;
  EXPECT_THROW(sdv::loadRigConfig(dir / "mixed.yaml"), std::runtime_error);

  // Refined extrinsics written elsewhere: the pose is replaced, relative paths still resolve.
  const Sophus::SE3d T(Sophus::SO3d::rotZ(0.01) * rig.cameras[0].T_b_c.so3(), Eigen::Vector3d(2.01, 0.1, 1.5));
  std::filesystem::create_directories(dir / "out");
  sdv::writeRigConfig(dir / "rig.yaml", {{"F_FISHEYE_C", T}}, dir / "out" / "refined.yaml");
  const sdv::RigConfig refined = sdv::loadRigConfig(dir / "out" / "refined.yaml");
  EXPECT_EQ(refined.aimRecord, dir / "record");
  EXPECT_DOUBLE_EQ(refined.cameras[0].camera.camera.fx, 80.0);
  EXPECT_LT((refined.cameras[0].T_b_c.matrix() - T.matrix()).cwiseAbs().maxCoeff(), 1e-9);

  // image_width shrinks the camera (pixel-centre convention) before any run-time scale.
  std::ofstream(dir / "scaled.yaml") << "aim_record:\n  record: record\n  calibration: config.zip\n"
                                        "cameras:\n  - name: F_FISHEYE_C\n    camera: front.yaml\n"
                                        "    image_width: 100\n" + pose;
  const sdv::Camera& half = sdv::loadRigConfig(dir / "scaled.yaml").cameras[0].camera.camera;
  EXPECT_EQ(half.width, 100);
  EXPECT_EQ(half.height, 50);
  EXPECT_DOUBLE_EQ(half.fx, 40.0);
  EXPECT_DOUBLE_EQ(half.cx, 49.5);
  EXPECT_DOUBLE_EQ(half.cy, 24.5);

  // mask_inflate grows the masked area (rig-wide, per camera overridden) but not from the image border.
  cv::Mat mask(100, 200, CV_8UC1, cv::Scalar(255));
  mask.rowRange(80, 100).setTo(0);
  cv::imwrite((dir / "mask.png").string(), mask);
  std::ofstream(dir / "masked.yaml") << "width: 200\nheight: 100\nfx: 80\nfy: 80\ncx: 99.5\ncy: 49.5\nmask: mask.png\n";
  std::ofstream(dir / "inflated.yaml") << "aim_record:\n  record: record\n  calibration: config.zip\nmask_inflate: 5\n"
                                          "cameras:\n  - name: A\n    camera: masked.yaml\n" + pose +
                                              "  - name: B\n    camera: masked.yaml\n    mask_inflate: 0\n" + pose;
  const sdv::RigConfig inflated = sdv::loadRigConfig(dir / "inflated.yaml");
  const cv::Mat& a = inflated.cameras[0].camera.mask;
  EXPECT_EQ(a.at<std::uint8_t>(74, 100), 255);
  EXPECT_EQ(a.at<std::uint8_t>(75, 100), 0);
  EXPECT_EQ(a.at<std::uint8_t>(0, 0), 255);
  EXPECT_EQ(inflated.cameras[1].camera.mask.at<std::uint8_t>(79, 100), 255);
}

TEST(CameraConfig, ScaleAndCropKeepProjectionsConsistent) {
  sdv::CameraConfig config;
  config.camera = sdv::Camera::eucm(300, 301, 330.2, 190.7, 0.6, 1.05, 660, 380);
  config.mask = circleMask(660, 380);
  const sdv::Camera cam = sdv::prepareCamera(config, 0.5, 16);
  EXPECT_EQ(cam.width, 320);  // 330 cropped to a multiple of 16
  EXPECT_EQ(cam.height, 176);  // 190 -> 176
  ASSERT_TRUE(cam.mask);
  EXPECT_EQ(cam.mask->width(0), 320);

  const Eigen::Vector3d p(0.4, -0.2, 1.0);
  Eigen::Vector2d full, half;
  ASSERT_TRUE(config.camera.project(p, full));
  ASSERT_TRUE(cam.project(p, half));
  EXPECT_NEAR(half.x(), (full.x() + 0.5) * 0.5 - 0.5, 1e-9);
  EXPECT_NEAR(half.y(), (full.y() + 0.5) * 0.5 - 0.5, 1e-9);

  const cv::Mat img(380, 660, CV_8UC3, cv::Scalar(10, 20, 30));
  const cv::Mat prepared = sdv::prepareImage(img, 0.5, cam);
  EXPECT_EQ(prepared.cols, 320);
  EXPECT_EQ(prepared.rows, 176);
  EXPECT_EQ(prepared.type(), CV_8UC3);
}

TEST(FrameSource, ImageFolderInNameOrderWithStride) {
  const auto dir = tempDir("image_folder");
  for (int i = 0; i < 7; ++i)
    cv::imwrite((dir / cv::format("%03d.png", i)).string(), cv::Mat(8, 8, CV_8UC1, cv::Scalar(10 * i)));
  sdv::SubsampledSource src(std::make_unique<sdv::ImageFolderSource>(dir), 1, 3);
  for (int expected : {1, 4}) {
    const cv::Mat m = src.next();
    ASSERT_FALSE(m.empty());
    EXPECT_EQ(m.at<uint8_t>(0, 0), 10 * expected);
  }
  EXPECT_TRUE(src.next().empty());
  src.rewind();
  EXPECT_EQ(src.next().at<uint8_t>(0, 0), 10);
}

TEST(FrameSource, VideoRoundTrip) {
  const auto file = tempDir("video") / "clip.avi";
  cv::VideoWriter writer(file.string(), cv::CAP_FFMPEG, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10.0, {64, 48});
  if (!writer.isOpened()) GTEST_SKIP() << "no FFmpeg video writer";
  for (int i = 0; i < 5; ++i) writer.write(cv::Mat(48, 64, CV_8UC3, cv::Scalar(40 * i, 40 * i, 40 * i)));
  writer.release();

  sdv::VideoSource video(file);
  int count = 0;
  for (cv::Mat f = video.next(); !f.empty(); f = video.next()) {
    EXPECT_EQ(f.cols, 64);
    EXPECT_NEAR(f.at<cv::Vec3b>(24, 32)[0], 40 * count, 6);
    ++count;
  }
  EXPECT_EQ(count, 5);
  video.rewind();
  EXPECT_FALSE(video.next().empty());
}
