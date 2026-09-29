#pragma once

#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <sophus/se3.hpp>

namespace sdv::aimrec {

// EUCM camera of an aiMotive sensorconfig (Khomutenko et al. 2016: fx, fy, cx, cy, alpha, beta).
struct CameraCalibration {
  std::string label;     // e.g. F_FISHEYE_C
  std::string deviceId;  // camera serial, names the recorded streams
  int width = 0, height = 0;
  double fx = 0, fy = 0, cx = 0, cy = 0, alpha = 0, beta = 1;
  // Maps camera coordinates (OpenCV: x right, y down, z forward) to vehicle coordinates (x forward, y left, z up).
  Sophus::SE3d T_vehicle_camera;
  std::string maskFile;  // obstruction mask, relative to the vehicle directory; empty if none
  cv::Mat mask;          // CV_8UC1 at camera resolution, 255 = valid; empty if none or not loaded
};

struct VehicleCalibration {
  std::string vehicle;
  std::vector<CameraCalibration> cameras;

  const CameraCalibration* find(const std::string& label) const;
};

// Sensorconfig extrinsics: position in metres, orientation (Rx(roll) Ry(pitch) Rz(yaw))^T of a camera whose OpenCV
// axes are first turned to the vehicle axes (optical axis along vehicle x); positive yaw turns right, negative pitch
// looks down.
Sophus::SE3d vehicleFromCamera(const Eigen::Vector3d& posMeter, const std::array<double, 3>& yawPitchRollDeg);

// EUCM cameras of a sensorconfig YAML (either a `sensors:` map or a bare sensor list); other sensors and camera
// models are skipped. Masks are not loaded.
std::vector<CameraCalibration> parseSensorConfig(const std::string& yaml, const std::string& name);

// Calibration from aimprototype_config.zip (vehicle_database/VEHICLES/<vehicle>/sensorconfig.yaml and its masks) or
// from a sensorconfig.yaml on disk (masks relative to its directory). An empty `vehicle` picks the only vehicle of
// the zip. Masks are loaded for the cameras in `maskLabels` (all without it).
VehicleCalibration loadCalibration(const std::filesystem::path& zipOrSensorConfig, const std::string& vehicle = {},
                                   const std::optional<std::vector<std::string>>& maskLabels = std::nullopt);

}  // namespace sdv::aimrec
