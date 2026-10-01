#include <sdv/point_filter.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_map>

#include <opencv2/imgproc.hpp>

#include <sdv/parallel.h>

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

FreeSpaceResult freeSpaceFloaters(const Rig& rig, const std::vector<Sophus::SE3d>& T_w_b,
                                  const std::vector<MapPoint>& points, const FreeSpaceSettings& settings) {
  FreeSpaceResult result;
  const size_t n = points.size();
  result.floater.assign(n, 0);
  result.support.assign(n, 0);
  result.through.assign(n, 0);

  auto hostKey = [](const MapPoint& p) { return static_cast<std::int64_t>(p.frameIndex) * 64 + p.camera; };
  std::map<std::int64_t, std::vector<std::uint32_t>> viewPoints;
  for (size_t i = 0; i < n; ++i)
    if (points[i].source == MapPointSource::SemiDense && points[i].frameIndex >= 0 &&
        points[i].frameIndex < static_cast<int>(T_w_b.size()) && points[i].camera < rig.size())
      viewPoints[hostKey(points[i])].push_back(static_cast<std::uint32_t>(i));
  std::vector<std::pair<std::int64_t, const std::vector<std::uint32_t>*>> views;
  for (const auto& [key, own] : viewPoints) views.emplace_back(key, &own);
  result.views = static_cast<int>(views.size());

  // Points by cell of the search radius: a view visits the cells around its camera.
  const double cell = std::max(settings.radius / 4, 0.5);
  auto cellOf = [&](const Eigen::Vector3d& x) {
    return std::array<long long, 3>{static_cast<long long>(std::floor(x.x() / cell)),
                                    static_cast<long long>(std::floor(x.y() / cell)),
                                    static_cast<long long>(std::floor(x.z() / cell))};
  };
  auto cellKey = [](long long x, long long y, long long z) {
    constexpr long long kOffset = 1 << 20;
    return (static_cast<std::uint64_t>(x + kOffset) & 0x1fffff) << 42 |
           (static_cast<std::uint64_t>(y + kOffset) & 0x1fffff) << 21 | (static_cast<std::uint64_t>(z + kOffset) & 0x1fffff);
  };
  std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> grid;
  for (size_t i = 0; i < n; ++i) {
    const auto c = cellOf(points[i].position);
    grid[cellKey(c[0], c[1], c[2])].push_back(static_cast<std::uint32_t>(i));
  }

  std::vector<std::atomic<int>> support(n), through(n);
  const int k = 2 * settings.window + 1;
  const cv::Mat kernel = cv::Mat::ones(k, k, CV_8U);
  const long long reach = static_cast<long long>(std::ceil(settings.radius / cell));
  parallelChunks(views.size(), [&](size_t, size_t begin, size_t end) {
    for (size_t vi = begin; vi < end; ++vi) {
      const std::int64_t key = views[vi].first;
      const int frame = static_cast<int>(key / 64), c = static_cast<int>(key % 64);
      const Camera& cam = rig.cameras[c];
      const Sophus::SE3d T_c_w = rig.T_c_b[c] * T_w_b[frame].inverse();
      cv::Mat depth(cam.height, cam.width, CV_32F, cv::Scalar(std::numeric_limits<float>::infinity()));
      for (std::uint32_t i : *views[vi].second) {
        const int u = static_cast<int>(std::lround(points[i].uv.x())), v = static_cast<int>(std::lround(points[i].uv.y()));
        if (u < 0 || v < 0 || u >= cam.width || v >= cam.height) continue;
        float& d = depth.at<float>(v, u);
        d = std::min(d, static_cast<float>(points[i].distance));
      }
      cv::erode(depth, depth, kernel);  // the nearest of the view's own points within the window

      const Eigen::Vector3d center = T_c_w.inverse().translation();
      const auto c0 = cellOf(center);
      for (long long dx = -reach; dx <= reach; ++dx)
        for (long long dy = -reach; dy <= reach; ++dy)
          for (long long dz = -reach; dz <= reach; ++dz) {
            const auto it = grid.find(cellKey(c0[0] + dx, c0[1] + dy, c0[2] + dz));
            if (it == grid.end()) continue;
            for (std::uint32_t i : it->second) {
              const MapPoint& p = points[i];
              if (p.source == MapPointSource::SemiDense && hostKey(p) == key) continue;
              if ((p.position - center).squaredNorm() > settings.radius * settings.radius) continue;
              const Eigen::Vector3d x = T_c_w * p.position;
              Eigen::Vector2d uv;
              if (!cam.project(x, uv)) continue;
              const int u = static_cast<int>(std::lround(uv.x())), v = static_cast<int>(std::lround(uv.y()));
              if (u < 0 || v < 0 || u >= cam.width || v >= cam.height) continue;
              const double dv = depth.at<float>(v, u);
              if (!std::isfinite(dv)) continue;
              const double dp = x.norm(), r = dv / dp;
              if (std::abs(r - 1) < settings.tolerance) support[i].fetch_add(1, std::memory_order_relaxed);
              else if (r > 1 + settings.margin && dv - dp > settings.minGap)
                through[i].fetch_add(1, std::memory_order_relaxed);
            }
          }
    }
  });
  for (size_t i = 0; i < n; ++i) {
    result.support[i] = support[i].load();
    result.through[i] = through[i].load();
    result.floater[i] = result.through[i] >= settings.minThrough && result.through[i] > result.support[i];
  }
  return result;
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
