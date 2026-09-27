#include <sdv/point_selector.h>

#include <algorithm>
#include <array>
#include <cmath>

namespace sdv {

std::vector<float> PointSelector::regionThresholds(const ImageLevel& img, const cv::Mat& mask, int& regionsX,
                                                  int& regionsY) const {
  const int R = m_settings.regionSize;
  regionsX = (img.width + R - 1) / R;
  regionsY = (img.height + R - 1) / R;
  constexpr int kBins = 50;

  // Regions without enough valid pixels get no median and are skipped when smoothing.
  constexpr float kNoMedian = -1.f;
  std::vector<float> medians(static_cast<size_t>(regionsX) * regionsY);
  for (int ry = 0; ry < regionsY; ++ry) {
    for (int rx = 0; rx < regionsX; ++rx) {
      std::array<int, kBins + 1> hist{};
      int count = 0;
      for (int v = ry * R; v < std::min((ry + 1) * R, img.height); ++v)
        for (int u = rx * R; u < std::min((rx + 1) * R, img.width); ++u) {
          if (!mask.empty() && mask.at<uint8_t>(v, u) == 0) continue;
          const int bin = std::min(static_cast<int>(std::sqrt(img.gradNormSq[v * img.width + u])), kBins);
          ++hist[bin];
          ++count;
        }
      if (count < R * R / 8) {
        medians[ry * regionsX + rx] = kNoMedian;
        continue;
      }
      int acc = 0, bin = 0;
      while (bin < kBins && acc + hist[bin] < count / 2) acc += hist[bin++];
      medians[ry * regionsX + rx] = static_cast<float>(bin) + 0.5f;
    }
  }

  // 3x3 smoothing avoids hard threshold jumps at region borders.
  std::vector<float> thresholds(medians.size());
  for (int ry = 0; ry < regionsY; ++ry)
    for (int rx = 0; rx < regionsX; ++rx) {
      float sum = 0;
      int n = 0;
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          const int x = rx + dx, y = ry + dy;
          if (x < 0 || y < 0 || x >= regionsX || y >= regionsY || medians[y * regionsX + x] == kNoMedian) continue;
          sum += medians[y * regionsX + x];
          ++n;
        }
      const float t = (n > 0 ? sum / n : 0.f) + m_settings.gradientOffset;
      thresholds[ry * regionsX + rx] = t * t;
    }
  return thresholds;
}

std::vector<Candidate> PointSelector::selectWithCellSize(const ImageLevel& img, const cv::Mat& mask, int cell,
                                                         const std::vector<float>& thresholds, int regionsX) const {
  const int R = m_settings.regionSize, b = m_settings.border;
  const int cellsX = (img.width + cell - 1) / cell, cellsY = (img.height + cell - 1) / cell;
  std::vector<uint8_t> occupied(static_cast<size_t>(cellsX) * cellsY, 0);
  std::vector<Candidate> out;

  for (int pass = 0; pass < m_settings.numPasses; ++pass) {
    const int span = 1 << pass;
    const int size = cell * span;
    const float factor = std::pow(m_settings.passFactor, 2.f * pass);
    for (int cy = 0; cy < cellsY; cy += span) {
      for (int cx = 0; cx < cellsX; cx += span) {
        bool taken = false;
        for (int y = cy; y < std::min(cy + span, cellsY) && !taken; ++y)
          for (int x = cx; x < std::min(cx + span, cellsX) && !taken; ++x) taken = occupied[y * cellsX + x] != 0;
        if (taken) continue;

        float best = 0;
        Eigen::Vector2i bestUv(-1, -1);
        for (int v = std::max(cy * cell, b); v < std::min(cy * cell + size, img.height - b); ++v) {
          for (int u = std::max(cx * cell, b); u < std::min(cx * cell + size, img.width - b); ++u) {
            const float g = img.gradNormSq[v * img.width + u];
            if (g <= best || g < factor * thresholds[(v / R) * regionsX + u / R]) continue;
            if (!mask.empty() && mask.at<uint8_t>(v, u) == 0) continue;
            best = g;
            bestUv = {u, v};
          }
        }
        if (bestUv.x() < 0) continue;
        out.push_back({bestUv, pass});
        occupied[(bestUv.y() / cell) * cellsX + bestUv.x() / cell] = 1;
      }
    }
  }
  return out;
}

std::vector<Candidate> PointSelector::select(const ImageLevel& img, const cv::Mat& mask) const {
  int regionsX = 0, regionsY = 0;
  const auto thresholds = regionThresholds(img, mask, regionsX, regionsY);

  const double area = mask.empty() ? double(img.width) * img.height : double(cv::countNonZero(mask));
  double cell = std::sqrt(area / m_settings.targetPoints);
  std::vector<Candidate> best;
  for (int it = 0; it < m_settings.adaptIterations; ++it) {
    const int c = std::max(1, static_cast<int>(std::lround(cell)));
    auto pts = selectWithCellSize(img, mask, c, thresholds, regionsX);
    const double ratio = double(pts.size()) / m_settings.targetPoints;
    if (best.empty() || std::abs(ratio - 1.0) < std::abs(double(best.size()) / m_settings.targetPoints - 1.0))
      best = std::move(pts);
    if (std::abs(ratio - 1.0) < 0.15 || ratio == 0.0) break;
    cell *= std::sqrt(ratio);
  }
  return best;
}

}  // namespace sdv
