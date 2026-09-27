#pragma once

#include <vector>

#include <opencv2/core.hpp>

namespace sdv {

// Valid image area per pyramid level (e.g. the fisheye image circle). Level l is valid where all 2^l x 2^l input
// pixels are valid, eroded so that bilinear samples and central-difference gradients never read invalid pixels.
class ValidityMask {
 public:
  static constexpr int kLevels = 8;

  // image: CV_8UC1, pixels above 128 are valid.
  explicit ValidityMask(const cv::Mat& image, int erosion = 2);

  int width(int level) const { return m_levels[level].cols; }
  int height(int level) const { return m_levels[level].rows; }
  bool valid(int level, int u, int v) const {
    const cv::Mat& m = m_levels[level];
    return u >= 0 && v >= 0 && u < m.cols && v < m.rows && m.at<uint8_t>(v, u) != 0;
  }
  // CV_8UC1, 255 = valid.
  const cv::Mat& level(int l) const { return m_levels[l]; }
  double validFraction(int level) const;

 private:
  std::vector<cv::Mat> m_levels;
};

}  // namespace sdv
