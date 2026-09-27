#include <vector>

#include <gtest/gtest.h>

#include <sdv/point_merge.h>

namespace {

sdv::MapPoint point(const Eigen::Vector3d& x, int frame, double distance, std::uint8_t grey) {
  sdv::MapPoint p{};
  p.position = x;
  p.intensity = grey;
  p.frameIndex = frame;
  p.camera = 0;
  p.uv = Eigen::Vector2d::Zero();
  p.distance = distance;
  p.observations = 3;
  p.relativeDepthSigma = 0.01;
  p.color = std::array<std::uint8_t, 3>{grey, grey, grey};
  return p;
}

// Two passes over a 10 cm grid of wall points, the second shifted by 1 cm; first-pass neighbours are 10 cm apart.
std::vector<sdv::MapPoint> twoPasses(int secondFrame) {
  std::vector<sdv::MapPoint> points;
  for (int i = 0; i < 20; ++i)
    for (int j = 0; j < 20; ++j) {
      const Eigen::Vector3d x(0.1 * i, 0.1 * j, 5.0);
      points.push_back(point(x, 10, 2.0, 100));
      points.push_back(point(x + Eigen::Vector3d(0.01, 0, 0), secondFrame, 4.0, 200));
    }
  return points;
}

TEST(PointMerge, MergesPassesWeightedByDistance) {
  size_t merged = 0;
  const auto out = sdv::mergeDuplicatePoints(twoPasses(500), {.maxDistance = 0.03, .minFrameGap = 100}, &merged);
  EXPECT_EQ(merged, 400u);
  ASSERT_EQ(out.size(), 400u);
  for (const auto& p : out) {
    EXPECT_EQ(p.frameIndex, 10);  // closer host kept
    EXPECT_EQ(p.observations, 6);
    // Weights 1/2^2 and 1/4^2: the merged point is 1/5 of the way to the far pass.
    EXPECT_NEAR(p.position.x() - 0.1 * std::round(p.position.x() / 0.1), 0.002, 1e-9);
    EXPECT_EQ((*p.color)[0], 120);
  }
}

TEST(PointMerge, RespectsFrameGapAndDistance) {
  size_t merged = 0;
  EXPECT_EQ(sdv::mergeDuplicatePoints(twoPasses(50), {.maxDistance = 0.03, .minFrameGap = 100}, &merged).size(), 800u);
  EXPECT_EQ(merged, 0u);
  EXPECT_EQ(sdv::mergeDuplicatePoints(twoPasses(500), {.maxDistance = 0.005, .minFrameGap = 100}, &merged).size(),
            800u);
  EXPECT_EQ(merged, 0u);
}

TEST(PointMerge, ThreePassesJoinInTwoRounds) {
  std::vector<sdv::MapPoint> points;
  for (int i = 0; i < 10; ++i) {
    const Eigen::Vector3d x(0.2 * i, 0, 5);
    points.push_back(point(x, 0, 2.0, 100));
    points.push_back(point(x + Eigen::Vector3d(0.005, 0, 0), 1000, 2.0, 100));
    points.push_back(point(x + Eigen::Vector3d(0.02, 0, 0), 2000, 2.0, 100));
  }
  EXPECT_EQ(sdv::mergeDuplicatePoints(points, {.maxDistance = 0.03, .minFrameGap = 100, .rounds = 2}).size(), 10u);
  EXPECT_EQ(sdv::mergeDuplicatePoints(points, {.maxDistance = 0.03, .minFrameGap = 100, .rounds = 1}).size(), 20u);
}

}  // namespace
