#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>

#include <gtest/gtest.h>

#include <sdv/io/ply.h>

TEST(Ply, WritesVerticesAndEdges) {
  std::vector<Sophus::SE3d> traj;
  for (int i = 0; i < 5; ++i) traj.emplace_back(Sophus::SO3d(), Eigen::Vector3d(i, 0, 0));

  sdv::PlyScene scene;
  scene.addTrajectory(traj, {255, 0, 0});
  scene.addPoint({1, 2, 3}, {0, 0, 255});
  scene.addCameraAxes(traj, 0.5, 2);
  EXPECT_EQ(scene.numVertices(), 5u + 1u + 3u * 6u);

  const auto file = std::filesystem::temp_directory_path() / "sdv_test_scene.ply";
  scene.write(file, false);
  std::string text;
  {
    std::ifstream in(file);
    std::stringstream ss;
    ss << in.rdbuf();
    text = ss.str();
  }
  EXPECT_NE(text.find("element vertex 24"), std::string::npos);
  EXPECT_NE(text.find("element edge 13"), std::string::npos);
  std::filesystem::remove(file);
}

TEST(Ply, SkipsNonFiniteVertices) {
  const double inf = std::numeric_limits<double>::infinity();
  sdv::PlyScene scene;
  EXPECT_TRUE(scene.addPoint({1, 2, 3}, {0, 0, 0}));
  EXPECT_FALSE(scene.addPoint({std::nan(""), 0, 0}, {0, 0, 0}));
  EXPECT_FALSE(scene.addPoint({0, inf, 0}, {0, 0, 0}));
  EXPECT_FALSE(scene.addPoint({1e39, 0, 0}, {0, 0, 0}));  // overflows float
  std::vector<Sophus::SE3d> traj = {Sophus::SE3d(), Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(inf, 0, 0)), Sophus::SE3d()};
  scene.addTrajectory(traj, {255, 0, 0});
  EXPECT_EQ(scene.numVertices(), 3u);
}
