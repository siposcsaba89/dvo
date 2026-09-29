#include <sdv/aimrec/calibration.h>

#include <algorithm>
#include <fstream>
#include <memory>
#include <numbers>
#include <sstream>
#include <stdexcept>

#include <Eigen/Geometry>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>
#include <zip.h>

namespace sdv::aimrec {

namespace {

struct ZipCloser {
  void operator()(zip_t* z) const { zip_discard(z); }
};
using ZipPtr = std::unique_ptr<zip_t, ZipCloser>;

ZipPtr openZip(const std::filesystem::path& file) {
  int err = 0;
  zip_t* z = zip_open(file.string().c_str(), ZIP_RDONLY, &err);
  if (!z) {
    zip_error_t e;
    zip_error_init_with_code(&e, err);
    const std::string msg = zip_error_strerror(&e);
    zip_error_fini(&e);
    throw std::runtime_error("cannot open " + file.string() + ": " + msg);
  }
  return ZipPtr(z);
}

std::optional<std::vector<std::uint8_t>> readZipEntry(zip_t* z, const std::string& entry) {
  zip_stat_t st;
  if (zip_stat(z, entry.c_str(), 0, &st) != 0 || !(st.valid & ZIP_STAT_SIZE)) return std::nullopt;
  zip_file_t* f = zip_fopen(z, entry.c_str(), 0);
  if (!f) return std::nullopt;
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(st.size));
  const zip_int64_t n = zip_fread(f, bytes.data(), bytes.size());
  zip_fclose(f);
  if (n != static_cast<zip_int64_t>(bytes.size())) throw std::runtime_error("cannot read zip entry " + entry);
  return bytes;
}

std::optional<std::vector<std::uint8_t>> readFile(const std::filesystem::path& file) {
  std::ifstream in(file, std::ios::binary);
  if (!in) return std::nullopt;
  return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(in), {});
}

template <typename T>
T required(const YAML::Node& node, const char* key, const std::string& where) {
  if (!node[key]) throw std::runtime_error(where + " lacks '" + key + "'");
  return node[key].as<T>();
}

template <std::size_t N>
std::array<double, N> vec(const YAML::Node& node, const char* key, const std::string& where) {
  const auto v = required<std::vector<double>>(node, key, where);
  if (v.size() != N) throw std::runtime_error(where + ": '" + key + "' needs " + std::to_string(N) + " values");
  std::array<double, N> a;
  std::copy(v.begin(), v.end(), a.begin());
  return a;
}

void attachMask(CameraCalibration& cam, const std::optional<std::vector<std::uint8_t>>& bytes) {
  if (!bytes) {
    spdlog::warn("{}: obstruction mask {} not found", cam.label, cam.maskFile);
    return;
  }
  cv::Mat mask = cv::imdecode(*bytes, cv::IMREAD_GRAYSCALE);
  if (mask.empty()) throw std::runtime_error(cam.label + ": cannot decode mask " + cam.maskFile);
  if (mask.cols != cam.width || mask.rows != cam.height) {
    spdlog::warn("{}: mask {}x{} resized to the camera resolution {}x{}", cam.label, mask.cols, mask.rows, cam.width,
                 cam.height);
    cv::resize(mask, mask, cv::Size(cam.width, cam.height), 0, 0, cv::INTER_NEAREST);
  }
  cam.mask = mask;
}

}  // namespace

const CameraCalibration* VehicleCalibration::find(const std::string& label) const {
  for (const auto& c : cameras)
    if (c.label == label) return &c;
  return nullptr;
}

Sophus::SE3d vehicleFromCamera(const Eigen::Vector3d& posMeter, const std::array<double, 3>& yawPitchRollDeg) {
  const double d = std::numbers::pi / 180.0;
  const Eigen::Matrix3d R_vehicle = (Eigen::AngleAxisd(yawPitchRollDeg[2] * d, Eigen::Vector3d::UnitX()) *
                                     Eigen::AngleAxisd(yawPitchRollDeg[1] * d, Eigen::Vector3d::UnitY()) *
                                     Eigen::AngleAxisd(yawPitchRollDeg[0] * d, Eigen::Vector3d::UnitZ()))
                                        .toRotationMatrix()
                                        .transpose();
  // OpenCV camera axes to vehicle axes: z (optical axis) -> x, x (right) -> -y, y (down) -> -z.
  const Eigen::Matrix3d swap = (Eigen::AngleAxisd(-std::numbers::pi / 2, Eigen::Vector3d::UnitZ()) *
                                Eigen::AngleAxisd(-std::numbers::pi / 2, Eigen::Vector3d::UnitX()))
                                   .toRotationMatrix();
  return Sophus::SE3d(Sophus::SO3d::fitToSO3(R_vehicle * swap), posMeter);
}

std::vector<CameraCalibration> parseSensorConfig(const std::string& yaml, const std::string& name) {
  YAML::Node root;
  try {
    root = YAML::Load(yaml);
  } catch (const YAML::Exception& e) {
    throw std::runtime_error("cannot parse " + name + ": " + e.what());
  }
  const YAML::Node sensors = root.IsMap() ? root["sensors"] : root;
  if (!sensors || !sensors.IsSequence()) throw std::runtime_error(name + " has no sensor list");

  std::vector<CameraCalibration> cameras;
  for (const YAML::Node& s : sensors) {
    if (!s["sensor_type"] || s["sensor_type"].as<std::string>() != "camera") continue;
    const std::string label = required<std::string>(s, "label", name);
    const std::string where = name + " camera " + label;
    const std::string model = s["model"] ? s["model"].as<std::string>() : "";
    if (model != "eucm") {
      spdlog::warn("{}: camera model '{}' is not supported, skipped", where, model);
      continue;
    }
    CameraCalibration c;
    c.label = label;
    c.deviceId = required<std::string>(s, "device_id", where);
    const auto res = vec<2>(s, "image_resolution_px", where);
    const auto f = vec<2>(s, "focal_length_px", where);
    const auto pp = vec<2>(s, "principal_point_px", where);
    c.width = static_cast<int>(res[0]);
    c.height = static_cast<int>(res[1]);
    c.fx = f[0];
    c.fy = f[1];
    c.cx = pp[0];
    c.cy = pp[1];
    c.alpha = required<double>(s, "alpha", where);
    c.beta = required<double>(s, "beta", where);
    const auto pos = vec<3>(s, "pos_meter", where);
    c.T_vehicle_camera = vehicleFromCamera({pos[0], pos[1], pos[2]}, vec<3>(s, "yaw_pitch_roll_deg", where));
    if (s["custom_vars"] && s["custom_vars"]["obstruction_mask_file"])
      c.maskFile = s["custom_vars"]["obstruction_mask_file"].as<std::string>();
    cameras.push_back(std::move(c));
  }
  return cameras;
}

VehicleCalibration loadCalibration(const std::filesystem::path& zipOrSensorConfig, const std::string& vehicle,
                                   const std::optional<std::vector<std::string>>& maskLabels) {
  VehicleCalibration calib;
  const std::string name = zipOrSensorConfig.string();
  auto wantsMask = [&](const CameraCalibration& c) {
    return !c.maskFile.empty() && (!maskLabels || std::ranges::find(*maskLabels, c.label) != maskLabels->end());
  };

  if (zipOrSensorConfig.extension() != ".zip") {
    const auto bytes = readFile(zipOrSensorConfig);
    if (!bytes) throw std::runtime_error("cannot open " + name);
    calib.vehicle = vehicle;
    calib.cameras = parseSensorConfig(std::string(bytes->begin(), bytes->end()), name);
    for (auto& c : calib.cameras)
      if (wantsMask(c)) attachMask(c, readFile(zipOrSensorConfig.parent_path() / c.maskFile));
    return calib;
  }

  // Any entry .../VEHICLES/<vehicle>/sensorconfig.yaml; its directory anchors the mask paths.
  const ZipPtr zip = openZip(zipOrSensorConfig);
  std::vector<std::pair<std::string, std::string>> found;  // (entry, vehicle)
  const zip_int64_t n = zip_get_num_entries(zip.get(), 0);
  for (zip_int64_t i = 0; i < n; ++i) {
    const std::string entry = zip_get_name(zip.get(), static_cast<zip_uint64_t>(i), 0);
    const std::filesystem::path p(entry);
    if (p.filename() != "sensorconfig.yaml" || p.parent_path().parent_path().filename() != "VEHICLES") continue;
    const std::string v = p.parent_path().filename().string();
    // Zips written by the recorder can list an entry twice.
    if ((vehicle.empty() || v == vehicle) && std::ranges::find(found, v, &decltype(found)::value_type::second) == found.end())
      found.emplace_back(entry, v);
  }
  if (found.empty())
    throw std::runtime_error(name + " has no VEHICLES/" + (vehicle.empty() ? "*" : vehicle) + "/sensorconfig.yaml");
  if (found.size() > 1) throw std::runtime_error(name + " holds several vehicles; name one");

  const auto& [entry, v] = found.front();
  const auto bytes = readZipEntry(zip.get(), entry);
  if (!bytes) throw std::runtime_error("cannot read " + entry + " from " + name);
  calib.vehicle = v;
  calib.cameras = parseSensorConfig(std::string(bytes->begin(), bytes->end()), name + ":" + entry);
  const std::string dir = std::filesystem::path(entry).parent_path().generic_string();
  for (auto& c : calib.cameras)
    if (wantsMask(c)) attachMask(c, readZipEntry(zip.get(), dir + "/" + c.maskFile));
  return calib;
}

}  // namespace sdv::aimrec
