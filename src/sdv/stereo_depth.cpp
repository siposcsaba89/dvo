#include <sdv/stereo_depth.h>

#include <cmath>
#include <stdexcept>

#include <opencv2/calib3d.hpp>

namespace sdv {

cv::Mat stereoInverseDistance(const cv::Mat& left, const cv::Mat& right, const Camera& cam, double baseline,
                              double minDisparity) {
  if (!cam.isPinhole()) throw std::invalid_argument("stereoInverseDistance requires rectified pinhole images");

  constexpr int kBlock = 5;
  auto sgbm = cv::StereoSGBM::create(0, 128, kBlock, 8 * kBlock * kBlock, 32 * kBlock * kBlock, 1, 63, 10, 100, 2,
                                     cv::StereoSGBM::MODE_SGBM_3WAY);
  cv::Mat disp16;
  sgbm->compute(left, right, disp16);

  cv::Mat rho(left.size(), CV_32F, cv::Scalar(0));
  for (int v = 0; v < disp16.rows; ++v) {
    const short* d = disp16.ptr<short>(v);
    float* out = rho.ptr<float>(v);
    const double my = (v - cam.cy) / cam.fy;
    for (int u = 0; u < disp16.cols; ++u) {
      const double disparity = d[u] / 16.0;
      if (disparity < minDisparity) continue;
      const double depth = cam.fx * baseline / disparity;
      const double mx = (u - cam.cx) / cam.fx;
      out[u] = static_cast<float>(1.0 / (depth * std::sqrt(mx * mx + my * my + 1.0)));
    }
  }
  return rho;
}

}  // namespace sdv
