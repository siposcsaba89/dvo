#include <sdv/undistort.h>

#include <opencv2/imgproc.hpp>

namespace sdv {

Camera virtualPinhole(const Camera& cam, double focalScale) {
  const double f = focalScale * cam.fx;
  return Camera::pinhole(f, f * cam.fy / cam.fx, 0.5 * (cam.width - 1), 0.5 * (cam.height - 1), cam.width,
                         cam.height);
}

UndistortMap makeUndistortMap(const Camera& src, const Camera& dst) {
  UndistortMap map{cv::Mat(dst.height, dst.width, CV_32F, cv::Scalar(-1)),
                   cv::Mat(dst.height, dst.width, CV_32F, cv::Scalar(-1))};
  for (int v = 0; v < dst.height; ++v)
    for (int u = 0; u < dst.width; ++u) {
      Eigen::Vector3d bearing;
      Eigen::Vector2d uv;
      if (!dst.unproject(Eigen::Vector2d(u, v), bearing) || !src.project(bearing, uv) ||
          !src.isInside(uv.x(), uv.y(), 0.0))
        continue;
      map.mapX.at<float>(v, u) = static_cast<float>(uv.x());
      map.mapY.at<float>(v, u) = static_cast<float>(uv.y());
    }
  return map;
}

cv::Mat undistort(const cv::Mat& image, const UndistortMap& map) {
  cv::Mat out;
  cv::remap(image, out, map.mapX, map.mapY, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
  return out;
}

}  // namespace sdv
