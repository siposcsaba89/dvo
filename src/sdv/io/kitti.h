#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <sophus/se3.hpp>

#include <sdv/camera.h>

namespace sdv {

// Crops right/bottom to a multiple of `sizeMultiple`, which keeps the principal point.
class KittiSequence {
 public:
  KittiSequence(const std::filesystem::path& sequenceDir, int sizeMultiple);

  size_t size() const { return m_timestamps.size(); }
  const Camera& camera() const { return m_camera; }
  double baseline() const { return m_baseline; }
  double timestamp(size_t i) const { return m_timestamps[i]; }

  // camIndex 0 = left (image_0), 1 = right (image_1).
  cv::Mat loadImage(size_t i, int camIndex = 0) const;

 private:
  std::filesystem::path m_dir;
  Camera m_camera;
  double m_baseline = 0;
  std::vector<double> m_timestamps;
};

// T_world_cam per line, 3x4 row-major.
std::vector<Sophus::SE3d> loadKittiPoses(const std::filesystem::path& file);
void saveKittiPoses(const std::filesystem::path& file, const std::vector<Sophus::SE3d>& poses);

}  // namespace sdv
