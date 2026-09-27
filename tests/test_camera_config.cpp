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
