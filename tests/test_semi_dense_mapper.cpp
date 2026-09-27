#include <algorithm>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

#include <sdv/semi_dense_mapper.h>

#include <synthetic_scene.h>

namespace {

constexpr int kW = 320, kH = 240;

sdv::Rig surroundRig() {
  const sdv::Camera cam = sdv::Camera::eucm(140, 140, 160.2, 119.7, 0.6, 1.1, kW, kH);
  return {{cam, cam, cam},
          {synthetic::room::cameraFromBody(0.0, {1.5, 0.0, 0.5}),
           synthetic::room::cameraFromBody(M_PI / 2, {0.5, 0.4, 0.5}),
           synthetic::room::cameraFromBody(-M_PI / 2, {0.5, -0.4, 0.5})}};
}

Sophus::SE3d bodyPose(int k) {
  return Sophus::SE3d::exp((Sophus::Vector6d() << 0.1 * k, 0, 0, 0, 0, 0.01 * k).finished());
}

}  // namespace

TEST(SemiDenseMapper, RecoversSurfacesFromKnownPoses) {
  const sdv::Rig rig = surroundRig();
  sdv::SemiDenseSettings settings;
  settings.pointsPerImage = 4000;
  settings.traceFrames = 20;
  sdv::SemiDenseMapper mapper(rig, settings);
  const std::vector<sdv::AffineBrightness> affine(rig.size());
  constexpr int kFrames = 25;
  for (int k = 0; k < kFrames; ++k) {
    std::vector<cv::Mat> images;
    for (int c = 0; c < rig.size(); ++c) {
      cv::Mat img = synthetic::room::render(rig.cameras[c], rig.T_c_b[c] * bodyPose(k).inverse());
      cv::GaussianBlur(img, img, cv::Size(0, 0), 1.0);
      images.push_back(img);
    }
    mapper.addFrame(k, images, bodyPose(k), affine, k % 5 == 0);
  }
  const auto points = mapper.finish();

  std::vector<int> perCamera(rig.size(), 0);
  std::vector<double> errors;
  for (const auto& p : points) {
    ASSERT_EQ(p.source, sdv::MapPointSource::SemiDense);
    ++perCamera[p.camera];
    const Eigen::Vector3d center = (bodyPose(p.frameIndex) * rig.T_c_b[p.camera].inverse()).translation();
    const Eigen::Vector3d ray = p.position - center;
    errors.push_back(std::abs(ray.norm() / synthetic::room::castRay(center, ray.normalized()) - 1.0));
  }
  for (int c = 0; c < rig.size(); ++c) EXPECT_GT(perCamera[c], 1500) << "camera " << c;
  ASSERT_FALSE(errors.empty());
  std::sort(errors.begin(), errors.end());
  EXPECT_LT(errors[errors.size() / 2], 0.005);
  EXPECT_LT(errors[errors.size() * 95 / 100], 0.03);
}

TEST(SemiDenseMapper, VoxelMergeKeepsOnePointPerVoxel) {
  const sdv::Rig rig = surroundRig();
  sdv::SemiDenseSettings settings;
  settings.pointsPerImage = 3000;
  settings.traceFrames = 10;
  settings.voxelSize = 0.1;
  settings.minVoxelHosts = 2;
  settings.thin = true;
  sdv::SemiDenseMapper mapper(rig, settings);
  const std::vector<sdv::AffineBrightness> affine(rig.size());
  for (int k = 0; k < 20; ++k) {
    std::vector<cv::Mat> images;
    for (int c = 0; c < rig.size(); ++c) {
      cv::Mat img = synthetic::room::render(rig.cameras[c], rig.T_c_b[c] * bodyPose(k).inverse());
      cv::GaussianBlur(img, img, cv::Size(0, 0), 1.0);
      images.push_back(img);
    }
    mapper.addFrame(k, images, bodyPose(k), affine, k % 4 == 0);
  }
  const auto points = mapper.finish();
  ASSERT_GT(points.size(), 300u);
  EXPECT_LT(mapper.stats().merged, mapper.stats().accepted);
  std::vector<std::array<long long, 3>> keys;
  for (const auto& p : points)
    keys.push_back({static_cast<long long>(std::floor(p.position.x() / 0.1)),
                    static_cast<long long>(std::floor(p.position.y() / 0.1)),
                    static_cast<long long>(std::floor(p.position.z() / 0.1))});
  std::sort(keys.begin(), keys.end());
  EXPECT_EQ(std::adjacent_find(keys.begin(), keys.end()), keys.end());
}
