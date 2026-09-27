#include <sdv/io/rig_config.h>

#include <cmath>
#include <stdexcept>

#include <Eigen/Geometry>
#include <yaml-cpp/yaml.h>

namespace sdv {

namespace {

std::filesystem::path resolve(const std::filesystem::path& base, const std::string& path) {
  const std::filesystem::path p(path);
  return p.is_relative() ? base / p : p;
}

Sophus::SE3d parsePose(const YAML::Node& node, const std::string& where) {
  if (!node || !node["translation"]) throw std::runtime_error(where + ": T_body_camera needs a translation");
  const auto t = node["translation"].as<std::vector<double>>();
  if (t.size() != 3) throw std::runtime_error(where + ": translation needs 3 values");
  Eigen::Matrix3d R;
  if (node["rotation_quaternion_wxyz"]) {
    const auto q = node["rotation_quaternion_wxyz"].as<std::vector<double>>();
    if (q.size() != 4) throw std::runtime_error(where + ": rotation_quaternion_wxyz needs 4 values");
    const Eigen::Quaterniond quat(q[0], q[1], q[2], q[3]);
    if (std::abs(quat.norm() - 1.0) > 1e-3) throw std::runtime_error(where + ": quaternion is not unit length");
    R = quat.normalized().toRotationMatrix();
  } else if (node["rotation_matrix"]) {
    const auto m = node["rotation_matrix"].as<std::vector<double>>();
    if (m.size() != 9) throw std::runtime_error(where + ": rotation_matrix needs 9 values");
    R << m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8];
    if ((R * R.transpose() - Eigen::Matrix3d::Identity()).norm() > 1e-3 || R.determinant() < 0)
      throw std::runtime_error(where + ": rotation_matrix is not a rotation");
  } else {
    throw std::runtime_error(where + ": T_body_camera needs rotation_quaternion_wxyz or rotation_matrix");
  }
  // Re-orthonormalise: configs usually carry a few printed decimals.
  const Eigen::Quaterniond q(R);
  return Sophus::SE3d(q.normalized(), Eigen::Vector3d(t[0], t[1], t[2]));
}

}  // namespace

RigConfig loadRigConfig(const std::filesystem::path& file) {
  YAML::Node root;
  try {
    root = YAML::LoadFile(file.string());
  } catch (const YAML::Exception& e) {
    throw std::runtime_error("cannot read rig config " + file.string() + ": " + e.what());
  }
  const std::filesystem::path base = file.parent_path();
  if (!root["cameras"] || !root["cameras"].IsSequence() || root["cameras"].size() == 0)
    throw std::runtime_error("rig config " + file.string() + " needs a non-empty 'cameras' list");

  RigConfig rig;
  for (size_t i = 0; i < root["cameras"].size(); ++i) {
    const YAML::Node node = root["cameras"][i];
    RigCameraConfig cam;
    cam.name = node["name"] ? node["name"].as<std::string>() : "cam" + std::to_string(i);
    const std::string where = "rig config " + file.string() + ", camera " + cam.name;
    if (!node["camera"]) throw std::runtime_error(where + ": missing 'camera'");
    if (node["camera"].IsScalar()) cam.camera = loadCameraConfig(resolve(base, node["camera"].as<std::string>()));
    else cam.camera = parseCameraConfig(node["camera"], file);
    cam.T_b_c = parsePose(node["T_body_camera"], where);
    if (node["video"]) cam.video = resolve(base, node["video"].as<std::string>());
    if (node["frame_offset"]) cam.frameOffset = node["frame_offset"].as<int>();
    if (cam.frameOffset < 0) throw std::runtime_error(where + ": frame_offset must not be negative");
    rig.cameras.push_back(std::move(cam));
  }
  return rig;
}

}  // namespace sdv
