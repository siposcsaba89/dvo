#include <gtest/gtest.h>

#include <sdv/point_selector.h>

#include <synthetic_scene.h>

namespace {

constexpr int kW = 320, kH = 240;

}  // namespace

TEST(PointSelector, HitsTargetCount) {
  const sdv::ImagePyramid pyr(synthetic::renderHost(kW, kH), 1);
  for (int target : {300, 800, 1500}) {
    sdv::PointSelectorSettings s;
    s.targetPoints = target;
    const auto pts = sdv::PointSelector(s).select(pyr.level(0));
    EXPECT_NEAR(double(pts.size()), target, 0.25 * target) << "target " << target;
  }
}

TEST(PointSelector, IgnoresFlatAreasBordersAndMask) {
  cv::Mat img = synthetic::renderHost(kW, kH);
  img(cv::Rect(0, 0, kW / 2, kH)).setTo(100.f);
  const sdv::ImagePyramid pyr(img, 1);
  cv::Mat mask(kH, kW, CV_8U, cv::Scalar(255));
  mask(cv::Rect(kW / 2, 0, kW / 4, kH / 2)).setTo(0);

  sdv::PointSelectorSettings s;
  s.targetPoints = 500;
  const auto pts = sdv::PointSelector(s).select(pyr.level(0), mask);
  ASSERT_FALSE(pts.empty());
  for (const auto& c : pts) {
    EXPECT_GE(c.uv.x(), kW / 2 - 1);
    EXPECT_NE(mask.at<uint8_t>(c.uv.y(), c.uv.x()), 0);
    EXPECT_GE(c.uv.x(), s.border);
    EXPECT_GE(c.uv.y(), s.border);
    EXPECT_LT(c.uv.x(), kW - s.border);
    EXPECT_LT(c.uv.y(), kH - s.border);
  }
}
