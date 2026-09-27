#include <sdv/validity_mask.h>

#include <algorithm>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace sdv {

ValidityMask::ValidityMask(const cv::Mat& image, int erosion) {
  if (image.empty() || image.type() != CV_8UC1) throw std::invalid_argument("validity mask must be CV_8UC1");
  cv::Mat raw = image > 128;
  const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * erosion + 1, 2 * erosion + 1));
  for (int l = 0; l < kLevels && raw.cols > 0 && raw.rows > 0; ++l) {
    cv::Mat eroded;
    // Outside the image counts as invalid, so the mask also keeps a border.
    cv::erode(raw, eroded, kernel, {-1, -1}, 1, cv::BORDER_CONSTANT, cv::Scalar(0));
    m_levels.push_back(eroded);
    cv::Mat coarse(raw.rows / 2, raw.cols / 2, CV_8UC1);
    for (int v = 0; v < coarse.rows; ++v)
      for (int u = 0; u < coarse.cols; ++u)
        coarse.at<uint8_t>(v, u) = std::min({raw.at<uint8_t>(2 * v, 2 * u), raw.at<uint8_t>(2 * v, 2 * u + 1),
                                             raw.at<uint8_t>(2 * v + 1, 2 * u),
                                             raw.at<uint8_t>(2 * v + 1, 2 * u + 1)});
    raw = coarse;
  }
}

double ValidityMask::validFraction(int level) const {
  const cv::Mat& m = m_levels[level];
  return m.empty() ? 0.0 : static_cast<double>(cv::countNonZero(m)) / m.total();
}

}  // namespace sdv
