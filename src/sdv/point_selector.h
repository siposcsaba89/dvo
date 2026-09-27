#pragma once

#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <sdv/image_pyramid.h>

namespace sdv {

struct PointSelectorSettings {
  int regionSize = 32;
  float gradientOffset = 7.f;  // DSO §3.2
  float passFactor = 0.75f;
  int numPasses = 3;
  int targetPoints = 2000;
  int adaptIterations = 4;
  int border = 4;
};

struct Candidate {
  Eigen::Vector2i uv;
  int pass;
};

class PointSelector {
 public:
  explicit PointSelector(PointSelectorSettings settings) : m_settings(settings) {}

  // mask: optional CV_8U, non-zero = usable.
  std::vector<Candidate> select(const ImageLevel& img, const cv::Mat& mask = {}) const;

 private:
  std::vector<float> regionThresholds(const ImageLevel& img, const cv::Mat& mask, int& regionsX, int& regionsY) const;
  std::vector<Candidate> selectWithCellSize(const ImageLevel& img, const cv::Mat& mask, int cell,
                                            const std::vector<float>& thresholds, int regionsX) const;

  PointSelectorSettings m_settings;
};

}  // namespace sdv
