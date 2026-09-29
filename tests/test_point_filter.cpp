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
