#pragma once

#include <cmath>
#include <limits>

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

// Closed textured box around the origin (vehicle frame: x forward, y left, z up), visible from any direction, for
// multi-camera rigs. Texture is defined on the two in-plane coordinates of each face, in metres.
namespace room {

inline const Eigen::Vector3d kMin(-8.0, -6.0, -1.5), kMax(8.0, 6.0, 2.5);

inline double texture(double a, double b) {
  return 128 + 50 * std::sin(a * 5.1) * std::cos(b * 6.3) + 30 * std::sin((a - 2 * b) * 3.7) +
         20 * std::cos((3 * a + b) * 2.3);
}

// Distance along the unit ray from `origin` (inside the box) to the first face, and the texture value there.
inline double castRay(const Eigen::Vector3d& origin, const Eigen::Vector3d& dir, double* value = nullptr) {
  double best = std::numeric_limits<double>::infinity();
  int axis = -1;
  for (int i = 0; i < 3; ++i) {
    if (std::abs(dir[i]) < 1e-12) continue;
    const double s = ((dir[i] > 0 ? kMax[i] : kMin[i]) - origin[i]) / dir[i];
    if (s > 0 && s < best) best = s, axis = i;
  }
  if (value && axis >= 0) {
    const Eigen::Vector3d p = origin + best * dir;
    *value = texture(p[(axis + 1) % 3], p[(axis + 2) % 3] + 10.0 * axis);
  }
  return best;
}

inline cv::Mat render(const sdv::Camera& cam, const Sophus::SE3d& T_c_w, double gain = 1.0, double offset = 0.0) {
  const Sophus::SE3d T_w_c = T_c_w.inverse();
  cv::Mat img(cam.height, cam.width, CV_32F, cv::Scalar(0));
  for (int v = 0; v < cam.height; ++v)
    for (int u = 0; u < cam.width; ++u) {
      Eigen::Vector3d b;
      if (!cam.unproject(Eigen::Vector2d(u, v), b)) continue;
      double value = 0;
      castRay(T_w_c.translation(), T_w_c.so3() * b, &value);
      img.at<float>(v, u) = static_cast<float>(gain * value + offset);
    }
  return img;
}

inline double trueRho(const Sophus::SE3d& T_c_w, const Eigen::Vector3d& bearing) {
  const Sophus::SE3d T_w_c = T_c_w.inverse();
  return 1.0 / castRay(T_w_c.translation(), T_w_c.so3() * bearing);
}

// T_c_b of a level OpenCV camera looking along body direction `yaw` (0 forward, pi/2 left), mounted at `position`.
inline Sophus::SE3d cameraFromBody(double yaw, const Eigen::Vector3d& position) {
  Eigen::Matrix3d R_b_c;
  R_b_c.col(0) = Eigen::Vector3d(std::sin(yaw), -std::cos(yaw), 0);
  R_b_c.col(1) = Eigen::Vector3d(0, 0, -1);
  R_b_c.col(2) = Eigen::Vector3d(std::cos(yaw), std::sin(yaw), 0);
  return Sophus::SE3d(R_b_c, position).inverse();
}

}  // namespace room

}  // namespace synthetic
