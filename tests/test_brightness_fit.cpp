#include <cmath>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

#include <sdv/brightness_fit.h>

#include <synthetic_scene.h>

namespace {

const sdv::Camera kCam = sdv::Camera::eucm(140, 140, 160.2, 119.7, 0.6, 1.1, 320, 240);

// Band-limited like a real image, so bilinear samples do not lose contrast.
cv::Mat renderBlurred(const Sophus::SE3d& T_c_w, double gain = 1.0, double offset = 0.0) {
  cv::Mat img = synthetic::room::render(kCam, T_c_w, gain, offset);
  cv::GaussianBlur(img, img, cv::Size(0, 0), 1.5);
  return img;
}

}  // namespace

TEST(BrightnessFit, RecoversAffineBrightnessDespiteOutliers) {
  const std::vector<Sophus::SE3d> T_c_w = {synthetic::room::cameraFromBody(0.0, {0.0, 0.0, 0.5}),
                                           synthetic::room::cameraFromBody(0.3, {0.4, 0.1, 0.5}),
                                           synthetic::room::cameraFromBody(0.6, {0.8, 0.2, 0.5})};
  const std::vector<sdv::AffineBrightness> truth = {{0.3, 12.0}, {-0.2, -5.0}, {0.0, 0.0}};
  std::vector<cv::Mat> images;
  for (size_t k = 0; k < T_c_w.size(); ++k)
    images.push_back(renderBlurred(T_c_w[k], std::exp(truth[k].a), truth[k].b));

  // Host pixels of another camera (brightness gauge: identity) at their true depth; every fifth irradiance is
  // wrong (occluded or mismatched).
  const Sophus::SE3d T_h_w = synthetic::room::cameraFromBody(0.2, {0.3, 0.0, 0.6});
  const cv::Mat host = renderBlurred(T_h_w);
  std::vector<sdv::IrradiancePoint> points;
  int i = 0;
  for (int v = 4; v < kCam.height - 4; v += 4)
    for (int u = 4; u < kCam.width - 4; u += 4, ++i) {
      Eigen::Vector3d bearing;
      if (!kCam.unproject(Eigen::Vector2d(u, v), bearing)) continue;
      const Eigen::Vector3d x = T_h_w.inverse() * (bearing / synthetic::room::trueRho(T_h_w, bearing));
      float value = host.at<float>(v, u);
      if (i % 5 == 0) value = 255 - value;
      points.push_back({x, value, i % 2});  // hosted by keyframes 0 and 1
    }
  sdv::BrightnessFitSettings settings;
  settings.hostWindow = 0;
  const auto fit = sdv::fitCameraBrightness(kCam, T_c_w, images, points, settings);
  // The same wall is blurred differently where the pixel scale differs (host and target view it from other image
  // positions): a few percent of contrast, as between real cameras.
  for (int k = 0; k < 2; ++k) {
    EXPECT_NEAR(fit.affine[k].a, truth[k].a, 0.05) << k;
    EXPECT_NEAR(fit.affine[k].b + std::exp(fit.affine[k].a) * 128, truth[k].b + std::exp(truth[k].a) * 128, 1.5) << k;
    EXPECT_GT(fit.inliers[k], 500) << k;
  }
  // Keyframe 2 hosts no point: it takes the nearest fitted keyframe.
  EXPECT_EQ(fit.inliers[2], 0);
  EXPECT_DOUBLE_EQ(fit.affine[2].a, fit.affine[1].a);
}
