#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <sdv/io/colmap.h>
#include <sdv/undistort.h>

namespace {

std::vector<std::string> dataLines(const std::filesystem::path& file) {
  std::ifstream in(file);
  std::vector<std::string> lines;
  for (std::string line; std::getline(in, line);)
    if (line.empty() || line[0] != '#') lines.push_back(line);
  return lines;
}

}  // namespace

TEST(Colmap, WritesConsistentTextModel) {
  const auto dir = std::filesystem::temp_directory_path() / "sdv_colmap_test";
  std::filesystem::remove_all(dir);
  const auto cam = sdv::Camera::pinhole(700, 710, 610.5, 180.25, 1232, 368);
  const Sophus::SE3d T0 = Sophus::SE3d::exp((Sophus::Vector6d() << 0.1, -0.2, 0.3, 0.01, 0.2, -0.05).finished());
  const Sophus::SE3d T1 = Sophus::SE3d::exp((Sophus::Vector6d() << -1.0, 0.0, 2.0, 0.0, -0.1, 0.0).finished());
  std::vector<sdv::ColmapImage> images = {{1, "000000.png", T0, {}}, {2, "000005.png", T1, {}}};
  std::vector<sdv::ColmapPoint> points = {{1, {1, 2, 3}, {10, 20, 30}}, {2, {-1, 0, 8}, {40, 50, 60}}};
  images[1].points2D.emplace_back(Eigen::Vector2d(100.25, 50.5), 2);
  points[1].track.emplace_back(2, 0);
  sdv::writeColmapText(dir, cam, images, points);

  const auto cameras = dataLines(dir / "cameras.txt");
  ASSERT_EQ(cameras.size(), 1u);
  EXPECT_EQ(cameras[0].rfind("1 PINHOLE 1232 368 700.", 0), 0u);

  const auto imageLines = dataLines(dir / "images.txt");
  ASSERT_EQ(imageLines.size(), 4u);
  EXPECT_TRUE(imageLines[1].empty());  // first image has no observations
  std::istringstream s(imageLines[2]);
  int id, camId;
  double qw, qx, qy, qz, tx, ty, tz;
  std::string name;
  s >> id >> qw >> qx >> qy >> qz >> tx >> ty >> tz >> camId >> name;
  EXPECT_EQ(id, 2);
  EXPECT_EQ(camId, 1);
  EXPECT_EQ(name, "000005.png");
  const Sophus::SE3d read(Eigen::Quaterniond(qw, qx, qy, qz).normalized(), Eigen::Vector3d(tx, ty, tz));
  EXPECT_LT((read * T1.inverse()).log().norm(), 1e-6);
  EXPECT_EQ(imageLines[3], "100.25 50.50 2");

  const auto pointLines = dataLines(dir / "points3D.txt");
  ASSERT_EQ(pointLines.size(), 2u);
  EXPECT_EQ(pointLines[1], "2 -1.0000 0.0000 8.0000 40 50 60 0.000 2 0");
  std::filesystem::remove_all(dir);
}

TEST(Colmap, RejectsFisheyeCamera) {
  const auto cam = sdv::Camera::eucm(140, 140, 160, 120, 0.6, 1.1, 320, 240);
  EXPECT_THROW(sdv::writeColmapText(std::filesystem::temp_directory_path() / "sdv_colmap_reject", cam, {}, {}),
               std::invalid_argument);
}

TEST(Undistort, MapsPinholePixelsToFisheyePixelsOfTheSameRay) {
  const auto fisheye = sdv::Camera::eucm(140, 140, 160.2, 119.7, 0.6, 1.1, 320, 240);
  const auto pinhole = sdv::virtualPinhole(fisheye, 0.8);
  ASSERT_TRUE(pinhole.isPinhole());
  const auto map = sdv::makeUndistortMap(fisheye, pinhole);
  for (const Eigen::Vector3d X : {Eigen::Vector3d(0.3, -0.2, 2.0), Eigen::Vector3d(-1.0, 0.5, 1.5)}) {
    Eigen::Vector2d uvPin, uvFish;
    ASSERT_TRUE(pinhole.project(X, uvPin) && fisheye.project(X, uvFish));
    const int u = static_cast<int>(std::lround(uvPin.x())), v = static_cast<int>(std::lround(uvPin.y()));
    Eigen::Vector3d b;
    ASSERT_TRUE(pinhole.unproject(Eigen::Vector2d(u, v), b));
    Eigen::Vector2d expected;
    ASSERT_TRUE(fisheye.project(b, expected));
    EXPECT_NEAR(map.mapX.at<float>(v, u), expected.x(), 1e-3);
    EXPECT_NEAR(map.mapY.at<float>(v, u), expected.y(), 1e-3);
    EXPECT_LT((expected - uvFish).norm(), 1.5);
  }
}
