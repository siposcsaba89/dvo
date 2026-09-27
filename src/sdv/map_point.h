#pragma once

#include <array>
#include <cstdint>
#include <optional>

#include <Eigen/Core>

namespace sdv {

enum class MapPointSource { Active, Candidate, SemiDense };

struct MapPoint {
  Eigen::Vector3d position;  // world
  float intensity;
  int frameIndex;  // input frame of the host image
  int camera;  // host camera
  Eigen::Vector2d uv;  // pixel in the host camera image
  double distance;  // from the host camera
  int observations;  // good residuals or matches (other cameras and frames)
  // Active: inverse-depth standard deviation relative to the inverse depth, for unit photometric noise, from the
  // window BA depth information (other parameters fixed). Candidate, SemiDense: half width of the traced
  // inverse-depth interval relative to the inverse depth.
  double relativeDepthSigma;
  MapPointSource source = MapPointSource::Active;
  std::optional<std::array<std::uint8_t, 3>> color;  // RGB, when the producer had a colour image
};

}  // namespace sdv
