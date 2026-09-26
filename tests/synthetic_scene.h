#pragma once

#include <cmath>

#include <opencv2/core.hpp>
#include <sophus/se3.hpp>

#include <sdv/camera.h>

namespace synthetic {

// Plane n.X = d in host coordinates; texture is defined on host pixel coordinates.
inline const Eigen::Vector3d kNormal = Eigen::Vector3d(0.1, -0.2, 1.0).normalized();
inline constexpr double kDist = 5.0;

inline double texture(double u, double v) {
  return 128 + 50 * std::sin(u / 7.0) * std::cos(v / 6.0) + 30 * std::sin((u - 2 * v) / 11.0) +
         20 * std::cos((3 * u + v) / 17.0);
}

inline cv::Mat renderHost(int w, int h) {
  cv::Mat img(h, w, CV_32F);
  for (int v = 0; v < h; ++v)
    for (int u = 0; u < w; ++u) img.at<float>(v, u) = static_cast<float>(texture(u, v));
  return img;
}

inline cv::Mat renderTarget(const sdv::Camera& cam, const Sophus::SE3d& T_t_h, double gain = 1.0,
                            double offset = 0.0) {
  const Sophus::SE3d T_h_t = T_t_h.inverse();
  cv::Mat img(cam.height, cam.width, CV_32F, cv::Scalar(0));
  for (int v = 0; v < cam.height; ++v)
    for (int u = 0; u < cam.width; ++u) {
      Eigen::Vector3d b;
      if (!cam.unproject(Eigen::Vector2d(u, v), b)) continue;
      const Eigen::Vector3d origin = T_h_t.translation();
      const Eigen::Vector3d dir = T_h_t.so3() * b;
      const double s = (kDist - kNormal.dot(origin)) / kNormal.dot(dir);
      if (s <= 0) continue;
      Eigen::Vector2d uvHost;
      if (!cam.project(Eigen::Vector3d(origin + s * dir), uvHost)) continue;
      img.at<float>(v, u) = static_cast<float>(gain * texture(uvHost.x(), uvHost.y()) + offset);
    }
  return img;
}

// Inverse distance of the plane along the host bearing; <= 0 if the ray misses it.
inline double trueRho(const Eigen::Vector3d& bearing) { return kNormal.dot(bearing) / kDist; }

// Same for a camera at T_c_w, with the world frame being the host frame the texture is defined in.
inline double trueRho(const Sophus::SE3d& T_c_w, const Eigen::Vector3d& bearing) {
  const Sophus::SE3d T_w_c = T_c_w.inverse();
  const double denom = kNormal.dot(T_w_c.so3() * bearing);
  if (std::abs(denom) < 1e-9) return 0.0;
  const double s = (kDist - kNormal.dot(T_w_c.translation())) / denom;
  return s > 0 ? 1.0 / s : 0.0;
}

}  // namespace synthetic
