#include <sdv/io/rig_config.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

#include <Eigen/Geometry>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

#include <sdv/aimrec/recording.h>

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
  if (const YAML::Node aim = root["aim_record"]) {
    if (!aim["record"] || !aim["calibration"])
      throw std::runtime_error("rig config " + file.string() + ": aim_record needs 'record' and 'calibration'");
    rig.aimRecord = resolve(base, aim["record"].as<std::string>());
    rig.aimCalibration = resolve(base, aim["calibration"].as<std::string>());
  }
  const int rigInflate = root["mask_inflate"] ? root["mask_inflate"].as<int>() : 0;
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
    if (node["image_width"]) {
      const int w = node["image_width"].as<int>();
      if (w <= 0 || w > cam.camera.camera.width)
        throw std::runtime_error(where + ": image_width must be in (0, " + std::to_string(cam.camera.camera.width) + "]");
      cam.imageScale = static_cast<double>(w) / cam.camera.camera.width;
    }
    if (cam.imageScale != 1.0) {
      Camera scaled = prepareCamera({cam.camera.camera, {}}, cam.imageScale, 1);
      if (!cam.camera.mask.empty())
        cv::resize(cam.camera.mask, cam.camera.mask, cv::Size(scaled.width, scaled.height), 0, 0, cv::INTER_AREA);
      cam.camera.camera = scaled;
    }
    const int inflate = node["mask_inflate"] ? node["mask_inflate"].as<int>() : rigInflate;
    if (inflate < 0) throw std::runtime_error(where + ": mask_inflate must not be negative");
    if (inflate > 0 && !cam.camera.mask.empty()) {
      const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * inflate + 1, 2 * inflate + 1));
      cv::Mat valid = cam.camera.mask > 128;
      cv::erode(valid, cam.camera.mask, kernel, {-1, -1}, 1, cv::BORDER_REPLICATE);
    }
    if (!rig.aimRecord.empty() && (!cam.video.empty() || cam.frameOffset != 0))
      throw std::runtime_error(where + ": video and frame_offset do not apply to an aim_record rig");
    rig.cameras.push_back(std::move(cam));
  }
  return rig;
}

void writeRigConfig(const std::filesystem::path& source,
                    const std::vector<std::pair<std::string, Sophus::SE3d>>& T_b_c, const std::filesystem::path& out,
                    const std::vector<std::pair<std::string, std::array<double, 6>>>& intrinsics) {
  YAML::Node root = YAML::LoadFile(source.string());
  const std::filesystem::path base = std::filesystem::absolute(source).parent_path();
  auto absolute = [&](YAML::Node node) {
    if (node && node.IsScalar()) node = resolve(base, node.as<std::string>()).lexically_normal().generic_string();
  };
  if (root["aim_record"]) absolute(root["aim_record"]["record"]), absolute(root["aim_record"]["calibration"]);
  for (YAML::Node cam : root["cameras"]) {
    if (cam["camera"].IsScalar()) absolute(cam["camera"]);
    else absolute(cam["camera"]["mask"]);
    absolute(cam["video"]);
    const std::string name = cam["name"] ? cam["name"].as<std::string>() : "";
    for (const auto& [n, k] : intrinsics) {
      if (n != name) continue;
      YAML::Node inline_ = cam["camera"].IsScalar() ? YAML::Clone(YAML::LoadFile(cam["camera"].as<std::string>()))
                                                    : YAML::Clone(cam["camera"]);
      if (cam["camera"].IsScalar() && inline_["mask"]) {
        const std::filesystem::path camFile = cam["camera"].as<std::string>();
        inline_["mask"] = resolve(camFile.parent_path(), inline_["mask"].as<std::string>()).lexically_normal().generic_string();
      }
      const char* keys[] = {"fx", "fy", "cx", "cy", "alpha", "beta"};
      for (int i = 0; i < 6; ++i) inline_[keys[i]] = k[i];
      cam["camera"] = inline_;
    }
    for (const auto& [n, T] : T_b_c) {
      if (n != name) continue;
      const Eigen::Matrix3d R = T.rotationMatrix();
      YAML::Node pose;
      pose["translation"] = std::vector<double>{T.translation().x(), T.translation().y(), T.translation().z()};
      std::vector<double> rowMajor;
      for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) rowMajor.push_back(R(r, c));
      pose["rotation_matrix"] = rowMajor;
      pose["translation"].SetStyle(YAML::EmitterStyle::Flow);
      pose["rotation_matrix"].SetStyle(YAML::EmitterStyle::Flow);
      cam["T_body_camera"] = pose;
    }
  }
  YAML::Emitter e;
  e.SetDoublePrecision(12);
  e << root;
  std::ofstream f(out);
  if (!f) throw std::runtime_error("cannot write " + out.string());
  f << "# Written from " << std::filesystem::absolute(source).generic_string() << "\n" << e.c_str() << "\n";
}

std::vector<RigStream> openRigStreams(const RigConfig& rig, const std::vector<std::string>& names,
                                      const std::vector<std::string>& syncNames) {
  std::vector<const RigCameraConfig*> selected;
  for (const auto& c : rig.cameras)
    if (names.empty() || std::find(names.begin(), names.end(), c.name) != names.end()) selected.push_back(&c);
  for (const auto& n : names)
    if (std::none_of(rig.cameras.begin(), rig.cameras.end(), [&](const RigCameraConfig& c) { return c.name == n; }))
      throw std::invalid_argument("no rig camera " + n);
  if (selected.empty()) throw std::invalid_argument("no rig camera selected");

  std::vector<RigStream> streams;
  if (!rig.aimRecord.empty()) {
    std::vector<std::string> labels;
    for (const auto* c : selected) labels.push_back(c->name);
    // One recording for all selected cameras: the synced frame ids depend on the selection. The rig cameras carry
    // their own masks.
    auto recording = std::make_shared<aimrec::Recording>(rig.aimRecord, rig.aimCalibration, labels, false);
    if (!syncNames.empty() && syncNames != labels)
      recording->selectFrames(aimrec::Recording(rig.aimRecord, rig.aimCalibration, syncNames, false).syncedFrameIds());
    for (size_t i = 0; i < selected.size(); ++i)
      streams.push_back(
          {selected[i], std::make_unique<AimRecordSource>(recording, static_cast<int>(i), selected[i]->imageScale)});
  } else {
    for (const auto* c : selected) {
      if (c->video.empty()) throw std::invalid_argument("rig camera " + c->name + " has no video");
      streams.push_back({c, std::make_unique<SubsampledSource>(std::make_unique<VideoSource>(c->video),
                                                               static_cast<size_t>(c->frameOffset), 1)});
    }
  }
  if (rig.aimRecord.empty())
    for (auto& s : streams)
      if (s.config->imageScale != 1.0)
        s.source = std::make_unique<ScaledSource>(std::move(s.source), s.config->imageScale);
  return streams;
}

}  // namespace sdv
