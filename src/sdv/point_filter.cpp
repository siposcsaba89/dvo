#include <sdv/point_filter.h>

#include <array>
#include <cmath>
#include <map>

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

}  // namespace sdv
