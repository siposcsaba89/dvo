#include <sdv/point_filter.h>

#include <array>
#include <cmath>
#include <map>
#include <stdexcept>

namespace sdv {

std::vector<char> hasNeighbours(const std::vector<Eigen::Vector3d>& points, double radius, int minNeighbours) {
  std::vector<char> keep(points.size(), 1);
  if (minNeighbours <= 0) return keep;
  auto key = [&](const Eigen::Vector3d& x) {
    return std::array<long long, 3>{static_cast<long long>(std::floor(x.x() / radius)),
                                    static_cast<long long>(std::floor(x.y() / radius)),
                                    static_cast<long long>(std::floor(x.z() / radius))};
  };
  std::map<std::array<long long, 3>, std::vector<size_t>> grid;
  for (size_t i = 0; i < points.size(); ++i) grid[key(points[i])].push_back(i);
  for (size_t i = 0; i < points.size(); ++i) {
    const auto k = key(points[i]);
    int count = 0;
    for (long long dx = -1; dx <= 1 && count < minNeighbours; ++dx)
      for (long long dy = -1; dy <= 1 && count < minNeighbours; ++dy)
        for (long long dz = -1; dz <= 1 && count < minNeighbours; ++dz) {
          const auto it = grid.find({k[0] + dx, k[1] + dy, k[2] + dz});
          if (it == grid.end()) continue;
          for (size_t j : it->second)
            if (j != i && (points[j] - points[i]).norm() < radius && ++count >= minNeighbours) break;
        }
    keep[i] = count >= minNeighbours;
  }
  return keep;
}

void voxelThin(std::vector<Eigen::Vector3d>& points, std::vector<std::array<std::uint8_t, 3>>& colors, double voxel) {
  if (colors.size() != points.size()) throw std::invalid_argument("one colour per point required");
  if (voxel <= 0) return;
  struct Cell {
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Vector3d color = Eigen::Vector3d::Zero();
    int count = 0;
  };
  std::map<std::array<long long, 3>, Cell> grid;
  for (size_t i = 0; i < points.size(); ++i) {
    Cell& c = grid[{static_cast<long long>(std::floor(points[i].x() / voxel)),
                    static_cast<long long>(std::floor(points[i].y() / voxel)),
                    static_cast<long long>(std::floor(points[i].z() / voxel))}];
    c.position += points[i];
    c.color += Eigen::Vector3d(colors[i][0], colors[i][1], colors[i][2]);
    ++c.count;
  }
  points.clear();
  colors.clear();
  for (const auto& [key, c] : grid) {
    points.push_back(c.position / c.count);
    const Eigen::Vector3d rgb = c.color / c.count;
    colors.push_back({static_cast<std::uint8_t>(std::lround(rgb.x())), static_cast<std::uint8_t>(std::lround(rgb.y())),
                      static_cast<std::uint8_t>(std::lround(rgb.z()))});
  }
}

}  // namespace sdv
