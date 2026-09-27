#pragma once

#include <vector>

#include <sophus/se3.hpp>

#include <sdv/camera.h>

namespace sdv {

// Cameras rigidly mounted on a body. Vehicle rigs use the body frame x forward, y left, z up; camera frames are
// OpenCV frames (z forward, x right, y down). Poses of the odometry are body poses; for a single camera or KITTI
// the body frame is the (left) camera frame.
struct Rig {
  std::vector<Camera> cameras;
  std::vector<Sophus::SE3d> T_c_b;  // body to camera

  static Rig mono(const Camera& cam) { return Rig{{cam}, {Sophus::SE3d()}}; }

  int size() const { return static_cast<int>(cameras.size()); }
  // Relative pose from camera `from` to camera `to` of the same body pose.
  Sophus::SE3d T_to_from(int to, int from) const { return T_c_b[to] * T_c_b[from].inverse(); }
};

}  // namespace sdv
