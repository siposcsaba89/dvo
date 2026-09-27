#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <sophus/se3.hpp>

#include <sdv/io/camera_config.h>

namespace sdv {

// YAML rig description, cameras rigidly mounted on the vehicle (body frame x forward, y left, z up; camera
// frames are OpenCV frames):
//   cameras:
//     - name: front
//       camera: front.yaml          # camera YAML (see camera_config.h), or the same keys inline
//       video: front.h264           # optional input stream
//       frame_offset: 0             # optional: frames to skip at the start of this stream to synchronise it
//       T_body_camera:              # maps camera coordinates to body coordinates
//         translation: [x, y, z]    # metres
//         rotation_quaternion_wxyz: [w, x, y, z]   # or rotation_matrix: [9 values, row-major]
// Relative paths are relative to the rig file.
struct RigCameraConfig {
  std::string name;
  CameraConfig camera;
  Sophus::SE3d T_b_c;
  std::filesystem::path video;
  int frameOffset = 0;
};

struct RigConfig {
  std::vector<RigCameraConfig> cameras;
};

RigConfig loadRigConfig(const std::filesystem::path& file);

}  // namespace sdv
