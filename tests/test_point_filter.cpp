#include <vector>

#include <gtest/gtest.h>

#include <sdv/point_filter.h>

TEST(PointFilter, VoxelThinAveragesPerVoxel) {
  std::vector<Eigen::Vector3d> points = {{0.01, 0.01, 0.01}, {0.03, 0.03, 0.03}, {0.11, 0.0, 0.0}};
  std::vector<std::array<std::uint8_t, 3>> colors = {{0, 0, 0}, {100, 50, 10}, {7, 7, 7}};
  sdv::voxelThin(points, colors, 0.1);
  ASSERT_EQ(points.size(), 2u);
  const size_t a = points[0].x() < 0.1 ? 0 : 1;
  EXPECT_TRUE(points[a].isApprox(Eigen::Vector3d(0.02, 0.02, 0.02)));
  EXPECT_EQ(colors[a], (std::array<std::uint8_t, 3>{50, 25, 5}));
  EXPECT_EQ(colors[1 - a], (std::array<std::uint8_t, 3>{7, 7, 7}));
}

TEST(PointFilter, VoxelBestKeepsMostObservedSelectedPoint) {
  auto point = [](Eigen::Vector3d x, int observations, double sigma) {
    return sdv::MapPoint{x, 0.f, 0, 0, Eigen::Vector2d::Zero(), 1.0, observations, sigma, sdv::MapPointSource::SemiDense};
  };
  const std::vector<sdv::MapPoint> points = {
      point({0.01, 0.01, 0.01}, 5, 0.01), point({0.02, 0.02, 0.02}, 9, 0.02), point({0.03, 0.01, 0.02}, 9, 0.01),
      point({0.04, 0.04, 0.04}, 20, 0.001), point({-0.01, 0.0, 0.0}, 1, 0.1), point({0.11, 0.0, 0.0}, 2, 0.1)};
  const std::vector<char> select = {1, 1, 1, 0, 1, 1};
  EXPECT_EQ(sdv::voxelBest(points, select, 0.1), (std::vector<char>{0, 0, 1, 0, 1, 1}));
  EXPECT_EQ(sdv::voxelBest(points, select, 0.0), select);
}

TEST(PointFilter, FreeSpaceFindsPointsInFrontOfAWall) {
  // Pinhole cameras on a line, all looking at a wall 6 m ahead (camera z); each hosts a grid of wall points, the
  // first camera also hosts points at 3 m that do not exist (the others see the wall through them).
  const sdv::Camera cam = sdv::Camera::eucm(200, 200, 159.5, 119.5, 0.0, 1.0, 320, 240);
  sdv::Rig rig{{cam}, {Sophus::SE3d()}};
  std::vector<Sophus::SE3d> T_w_b;
  for (int f = 0; f < 6; ++f) T_w_b.push_back(Sophus::SE3d(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0.4 * f, 0, 0)));
  std::vector<sdv::MapPoint> points;
  auto add = [&](int f, int u, int v, double depth) {
    Eigen::Vector3d bearing;
    ASSERT_TRUE(cam.unproject(Eigen::Vector2d(u, v), bearing));
    const Eigen::Vector3d x = bearing * (depth / bearing.z());
    sdv::MapPoint p{T_w_b[f] * x, 0.f, f, 0, Eigen::Vector2d(u, v), x.norm(), 5, 0.01, sdv::MapPointSource::SemiDense};
    points.push_back(p);
  };
  for (int f = 0; f < 6; ++f)
    for (int v = 4; v < 236; v += 2)
      for (int u = 4; u < 316; u += 2) add(f, u, v, 6.0);
  const size_t wall = points.size();
  for (int v = 100; v < 140; v += 4)
    for (int u = 140; u < 180; u += 4) add(0, u, v, 3.0);

  const auto result = sdv::freeSpaceFloaters(rig, T_w_b, points);
  EXPECT_EQ(result.views, 6);
  size_t wallFloaters = 0, found = 0;
  for (size_t i = 0; i < wall; ++i) wallFloaters += result.floater[i];
  for (size_t i = wall; i < points.size(); ++i) found += result.floater[i];
  EXPECT_EQ(wallFloaters, 0u);
  EXPECT_EQ(found, points.size() - wall);
}
