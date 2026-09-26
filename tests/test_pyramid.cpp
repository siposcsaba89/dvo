#include <gtest/gtest.h>

#include <sdv/camera.h>
#include <sdv/image_pyramid.h>

namespace {

cv::Mat makeRamp(int w, int h, float a, float b, float c) {
  cv::Mat img(h, w, CV_32F);
  for (int v = 0; v < h; ++v)
    for (int u = 0; u < w; ++u) img.at<float>(v, u) = a * u + b * v + c;
  return img;
}

}  // namespace

TEST(ImagePyramid, RampGradientsAndInterpolation) {
  const float a = 0.7f, b = -0.3f, c = 100.f;
  const sdv::ImagePyramid pyr(makeRamp(64, 32, a, b, c), 3);
  ASSERT_EQ(pyr.numLevels(), 3);

  for (int l = 0; l < 3; ++l) {
    const auto& lvl = pyr.level(l);
    EXPECT_EQ(lvl.width, 64 >> l);
    EXPECT_EQ(lvl.height, 32 >> l);
    const float s = static_cast<float>(1 << l);
    const auto& p = lvl.at(lvl.width / 2, lvl.height / 2);
    EXPECT_NEAR(p[1], a * s, 1e-4);
    EXPECT_NEAR(p[2], b * s, 1e-4);
    const Eigen::Vector3f q = lvl.interpolate(3.25f, 2.5f);
    EXPECT_NEAR(q[0], lvl.at(3, 2)[0] + 0.25f * a * s + 0.5f * b * s, 1e-3);
  }
}

TEST(ImagePyramid, CoarseLevelConsistentWithCameraModel) {
  const float a = 0.5f, b = 0.25f;
  const sdv::ImagePyramid pyr(makeRamp(64, 64, a, b, 0.f), 3);
  const auto cam = sdv::Camera::eucm(50, 50, 31.3, 30.7, 0.6, 1.1, 64, 64);
  const Eigen::Vector3d X(0.1, -0.2, 2.0);
  Eigen::Vector2d uv0;
  ASSERT_TRUE(cam.project(X, uv0));
  const float i0 = a * static_cast<float>(uv0.x()) + b * static_cast<float>(uv0.y());
  for (int l = 1; l < 3; ++l) {
    Eigen::Vector2d uv;
    ASSERT_TRUE(cam.atLevel(l).project(X, uv));
    EXPECT_NEAR(pyr.level(l).interpolateIntensity(float(uv.x()), float(uv.y())), i0, 1e-3) << "level " << l;
  }
}

TEST(ImagePyramid, RejectsBadSize) {
  EXPECT_THROW(sdv::ImagePyramid(cv::Mat(30, 30, CV_32F, cv::Scalar(0)), 3), std::invalid_argument);
}
