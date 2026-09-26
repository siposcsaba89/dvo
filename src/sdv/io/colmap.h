#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <sophus/se3.hpp>

#include <sdv/camera.h>

namespace sdv {

struct ColmapImage {
  int id;
  std::string name;
  Sophus::SE3d T_c_w;
  std::vector<std::pair<Eigen::Vector2d, std::int64_t>> points2D;  // pixel, point3D id (-1 = none)
};

struct ColmapPoint {
  std::int64_t id;
  Eigen::Vector3d position;
  std::array<std::uint8_t, 3> rgb;
  double error = 0;
  std::vector<std::pair<int, int>> track;  // image id, index into that image's points2D
};

// Writes cameras.txt, images.txt and points3D.txt (COLMAP text model) for a single PINHOLE camera with id 1.
void writeColmapText(const std::filesystem::path& dir, const Camera& cam, const std::vector<ColmapImage>& images,
                     const std::vector<ColmapPoint>& points);

}  // namespace sdv
