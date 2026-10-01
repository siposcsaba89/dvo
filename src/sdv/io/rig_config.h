#pragma once

#include <array>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <sophus/se3.hpp>

#include <sdv/io/camera_config.h>
#include <sdv/io/frame_source.h>

namespace sdv {

// YAML rig description, cameras rigidly mounted on the vehicle (body frame x forward, y left, z up; camera
// frames are OpenCV frames):
//   aim_record:                     # optional: frames come from an aiMotive recording, camera names are its labels
//     record: path/record           #   directory with Camera_*_meta.dat / _data.h264
//     calibration: config.zip       #   aimprototype_config.zip or sensorconfig.yaml (finds the camera streams)
//   mask_inflate: 0                 # optional: masked areas grow by this many pixels (of the camera image after
//                                   #   image_width), e.g. a margin around the ego vehicle
//   cameras:
//     - name: front
//       camera: front.yaml          # camera YAML (see camera_config.h), or the same keys inline
//       video: front.h264           # input stream without aim_record
//       frame_offset: 0             # optional, video only: frames to skip at the start to synchronise it
//       image_width: 960            # optional: images (and camera) are resized to this width, keeping the aspect
//                                   #   ratio, before anything else (cameras of different resolution in one rig)
//       mask_inflate: 0             # optional: overrides the rig-wide value
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
  double imageScale = 1.0;  // from image_width; `camera` is already scaled
};

struct RigConfig {
  std::vector<RigCameraConfig> cameras;
  std::filesystem::path aimRecord, aimCalibration;  // empty without aim_record
};

RigConfig loadRigConfig(const std::filesystem::path& file);

// Copy of the rig file `source` with T_body_camera replaced for the named cameras (e.g. refined extrinsics); relative
// paths are made absolute so that `out` can live anywhere.
// Intrinsics (fx, fy, cx, cy, alpha, beta) for the named cameras, at the resolution of their camera YAML, replace
// those of the YAML (written inline).
void writeRigConfig(const std::filesystem::path& source,
                    const std::vector<std::pair<std::string, Sophus::SE3d>>& T_b_c, const std::filesystem::path& out,
                    const std::vector<std::pair<std::string, std::array<double, 6>>>& intrinsics = {});

struct RigStream {
  const RigCameraConfig* config;
  std::unique_ptr<FrameSource> source;  // synchronised run frame 0, 1, 2, ...
};

// Input streams of the selected cameras (all when `names` is empty), in rig order. Videos start at their frame
// offset; the cameras of an aiMotive recording run over the frame ids all `syncNames` cameras recorded (default: the
// selected ones), so a different selection can be given the frames of a run.
std::vector<RigStream> openRigStreams(const RigConfig& rig, const std::vector<std::string>& names,
                                      const std::vector<std::string>& syncNames = {});

}  // namespace sdv
