#pragma once

#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>

namespace sdv {

struct ImageLevel {
  int width = 0, height = 0;
  std::vector<Eigen::Vector3f> data;  // (I, dI/du, dI/dv), interleaved for one-lookup interpolation
  std::vector<float> gradNormSq;

  const Eigen::Vector3f& at(int u, int v) const { return data[v * width + u]; }

  // Requires 0 <= u < width-1, 0 <= v < height-1.
  Eigen::Vector3f interpolate(float u, float v) const {
    const int iu = static_cast<int>(u);
    const int iv = static_cast<int>(v);
    const float du = u - iu, dv = v - iv;
    const Eigen::Vector3f* p = &data[iv * width + iu];
    return (1 - dv) * ((1 - du) * p[0] + du * p[1]) +
           dv * ((1 - du) * p[width] + du * p[width + 1]);
  }

  float interpolateIntensity(float u, float v) const {
    const int iu = static_cast<int>(u);
    const int iv = static_cast<int>(v);
    const float du = u - iu, dv = v - iv;
    const Eigen::Vector3f* p = &data[iv * width + iu];
    return (1 - dv) * ((1 - du) * p[0][0] + du * p[1][0]) +
           dv * ((1 - du) * p[width][0] + du * p[width + 1][0]);
  }
};

class ImagePyramid {
 public:
  // CV_32FC1 input; size must be divisible by 2^(levels-1).
  ImagePyramid(const cv::Mat& image, int levels);

  int numLevels() const { return static_cast<int>(m_levels.size()); }
  const ImageLevel& level(int l) const { return m_levels[l]; }

 private:
  std::vector<ImageLevel> m_levels;
};

cv::Mat toFloatGray(const cv::Mat& image);

}  // namespace sdv
