#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>

#include <Eigen/Core>
#include <sophus/se3.hpp>

namespace sdv {

using Rgb = std::array<uint8_t, 3>;

// Coloured points plus line edges; trajectories become polylines. Non-finite vertices are skipped.
class PlyScene {
 public:
  bool addPoint(const Eigen::Vector3d& p, const Rgb& color);
  void addTrajectory(const std::vector<Sophus::SE3d>& T_w_c, const Rgb& color);
  // RGB = xyz camera axes of every `stride`-th pose.
  void addCameraAxes(const std::vector<Sophus::SE3d>& T_w_c, double length, size_t stride);

  size_t numVertices() const { return m_rgb.size(); }
  void write(const std::filesystem::path& file, bool binary = true) const;

 private:
  int32_t addVertex(const Eigen::Vector3d& p, const Rgb& color);
  void addEdge(int32_t a, int32_t b) { m_edges.insert(m_edges.end(), {a, b}); }

  std::vector<Eigen::Vector3f> m_xyz;
  std::vector<Rgb> m_rgb;
  std::vector<int32_t> m_edges;
};

}  // namespace sdv
