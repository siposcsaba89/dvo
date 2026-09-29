#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <sophus/se3.hpp>

#include <sdv/camera.h>
#include <sdv/io/colmap.h>
#include <sdv/photometric.h>
#include <sdv/undistort.h>

namespace sdv {

struct ColmapExportCamera {
  std::string name;  // image subfolder; empty = images directly in images/
  Camera camera;  // of the images passed to addFrame, with its validity mask
  Sophus::SE3d T_c_b;
};

struct ColmapExportSettings {
  double focalScale = 1.0;  // virtual pinhole of a non-pinhole camera (see virtualPinhole)
  // Virtual pinholes are cropped (focal length kept) to at most this field of view per axis; 0 = off. Towards
  // wide angles a pinhole stretches the source pixels (a fisheye at 66 deg off-axis by 4.5x).
  double maxFovDeg = 0.0;
  std::string imageExtension = ".png";
  bool correctBrightness = false;  // write e^-a (I - b) instead of the input images
};

struct ColmapExportPoint {
  Eigen::Vector3d position;
  std::array<std::uint8_t, 3> rgb;
  int frameIndex = -1, camera = -1;  // host image, for a one-image track (optional)
};

// COLMAP text model for GS / NeuS trainers: images/<camera>/<frame>.<ext> undistorted to pinholes, masks/ of the
// same names from the validity masks, sparse/0 with PINHOLE cameras, and exposure.txt (image, a, b) when frames
// come with brightness.
class ColmapExporter {
 public:
  ColmapExporter(const std::filesystem::path& root, std::vector<ColmapExportCamera> cameras,
                 ColmapExportSettings settings = {});

  // images: one per camera (grey or BGR, 8 or 16 bit), prepared for the camera; T_w_b: body pose; brightness:
  // empty or one per camera.
  void addFrame(int frameIndex, const std::vector<cv::Mat>& images, const Sophus::SE3d& T_w_b,
                const std::vector<AffineBrightness>& brightness = {});
  void write(const std::vector<ColmapExportPoint>& points);

  size_t numImages() const { return m_images.size(); }

 private:
  struct Output {
    ColmapExportCamera input;
    Camera pinhole;
    UndistortMap map;
    cv::Mat mask;
  };

  std::filesystem::path m_root;
  ColmapExportSettings m_settings;
  std::vector<Output> m_outputs;
  std::vector<ColmapImage> m_images;
  std::map<std::pair<int, int>, size_t> m_imageOf;  // (frame, camera)
  std::vector<std::pair<std::string, AffineBrightness>> m_exposure;
};

}  // namespace sdv
