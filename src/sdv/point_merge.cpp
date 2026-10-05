#include <sdv/point_merge.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <execution>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace sdv {

namespace {

std::uint64_t cellKey(long long x, long long y, long long z) {
  constexpr long long kOffset = 1 << 20;
  return (static_cast<std::uint64_t>(x + kOffset) & 0x1fffff) << 42 |
         (static_cast<std::uint64_t>(y + kOffset) & 0x1fffff) << 21 | (static_cast<std::uint64_t>(z + kOffset) & 0x1fffff);
}

MapPoint mergePair(const MapPoint& a, const MapPoint& b) {
  const double wa = 1.0 / std::max(a.distance * a.distance, 1e-12), wb = 1.0 / std::max(b.distance * b.distance, 1e-12);
  MapPoint m = wa >= wb ? a : b;
  m.position = (wa * a.position + wb * b.position) / (wa + wb);
  m.intensity = static_cast<float>((wa * a.intensity + wb * b.intensity) / (wa + wb));
  m.observations = a.observations + b.observations;
  m.relativeDepthSigma = std::min(a.relativeDepthSigma, b.relativeDepthSigma);
  if (a.color && b.color) {
    std::array<std::uint8_t, 3> c;
    for (int i = 0; i < 3; ++i) c[i] = static_cast<std::uint8_t>(std::lround((wa * (*a.color)[i] + wb * (*b.color)[i]) / (wa + wb)));
    m.color = c;
  } else if (!m.color) {
    m.color = a.color ? a.color : b.color;
  }
  return m;
}

}  // namespace

std::vector<MapPoint> mergeDuplicatePoints(std::vector<MapPoint> points, const PointMergeSettings& settings,
                                           size_t* merged) {
  size_t total = 0;
  const double r = settings.maxDistance;
  for (int round = 0; round < settings.rounds && r > 0 && points.size() > 1; ++round) {
    auto cell = [&](const Eigen::Vector3d& x) {
      return std::array<long long, 3>{static_cast<long long>(std::floor(x.x() / r)),
                                      static_cast<long long>(std::floor(x.y() / r)),
                                      static_cast<long long>(std::floor(x.z() / r))};
    };
    // Cells as a sorted list (16 bytes per point; a hash map of vectors took ~120 on long runs).
    std::vector<std::pair<std::uint64_t, std::uint32_t>> grid(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
      const auto c = cell(points[i].position);
      grid[i] = {cellKey(c[0], c[1], c[2]), static_cast<std::uint32_t>(i)};
    }
    std::sort(std::execution::par, grid.begin(), grid.end());
    constexpr size_t kNone = std::numeric_limits<size_t>::max();
    std::vector<size_t> nearest(points.size(), kNone);
    std::vector<size_t> order(points.size());
    std::iota(order.begin(), order.end(), 0);
    std::for_each(std::execution::par, order.begin(), order.end(), [&](size_t i) {
      const auto c = cell(points[i].position);
      double best = r * r;
      for (long long dx = -1; dx <= 1; ++dx)
        for (long long dy = -1; dy <= 1; ++dy)
          for (long long dz = -1; dz <= 1; ++dz) {
            const std::uint64_t key = cellKey(c[0] + dx, c[1] + dy, c[2] + dz);
            auto it = std::lower_bound(grid.begin(), grid.end(), std::pair<std::uint64_t, std::uint32_t>{key, 0});
            for (; it != grid.end() && it->first == key; ++it) {
              const size_t j = it->second;
              if (j == i) continue;
              if (points[j].frameIndex == points[i].frameIndex && points[j].camera == points[i].camera) continue;
              if (std::abs(points[j].frameIndex - points[i].frameIndex) < settings.minFrameGap) continue;
              const double d2 = (points[j].position - points[i].position).squaredNorm();
              if (d2 < best) best = d2, nearest[i] = j;
            }
          }
    });
    grid = {};
    order = {};
    // In place: a pair is written at or before its first point, the second one is still unread.
    size_t count = 0, w = 0;
    for (size_t i = 0; i < points.size(); ++i) {
      const size_t j = nearest[i];
      if (j != kNone && nearest[j] == i) {
        if (i < j) {
          MapPoint m = mergePair(points[i], points[j]);
          points[w++] = std::move(m);
          ++count;
        }
      } else {
        if (w != i) points[w] = std::move(points[i]);
        ++w;
      }
    }
    points.resize(w);
    total += count;
    if (count == 0) break;
  }
  if (merged) *merged = total;
  return points;
}

}  // namespace sdv
