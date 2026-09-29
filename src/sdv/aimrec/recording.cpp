#include <sdv/aimrec/recording.h>

#include <algorithm>
#include <future>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <utility>

#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

namespace sdv::aimrec {

namespace {

namespace fs = std::filesystem;

std::optional<std::pair<fs::path, fs::path>> findStreams(const fs::path& dir, const std::string& deviceId) {
  const std::string metaSuffix = "_" + deviceId + "_meta.dat";
  for (const auto& e : fs::directory_iterator(dir)) {
    const std::string name = e.path().filename().string();
    if (!e.is_regular_file() || !name.starts_with("Camera_") || !name.ends_with(metaSuffix)) continue;
    const fs::path data = dir / (name.substr(0, name.size() - std::string("meta.dat").size()) + "data.h264");
    if (fs::exists(data)) return std::pair{e.path(), data};
  }
  return std::nullopt;
}

std::string sessionVehicle(const fs::path& recordDir) {
  const fs::path session = recordDir / "00_session.yml";
  if (!fs::exists(session)) return {};
  try {
    const YAML::Node node = YAML::LoadFile(session.string());
    if (node["session"] && node["session"]["vehiclename"]) return node["session"]["vehiclename"].as<std::string>();
  } catch (const YAML::Exception& e) {
    spdlog::warn("cannot parse {}: {}", session.string(), e.what());
  }
  return {};
}

}  // namespace

Recording::Recording(const fs::path& recordDir, const fs::path& calibration, const std::vector<std::string>& labels,
                     bool loadMasks) {
  if (!fs::is_directory(recordDir)) throw std::runtime_error("no record directory " + recordDir.string());
  // "record/" and "record" name the same directory; the calibration zip sits next to it.
  const fs::path dir = recordDir.filename().empty() ? recordDir.parent_path() : recordDir;
  const fs::path calibFile = calibration.empty() ? dir.parent_path() / "aimprototype_config.zip" : calibration;
  std::optional<std::vector<std::string>> maskLabels;
  if (!loadMasks) maskLabels.emplace();
  else if (!labels.empty()) maskLabels = labels;
  VehicleCalibration calib = loadCalibration(calibFile, sessionVehicle(dir), maskLabels);
  m_vehicle = calib.vehicle;
  m_recordDir = dir;
  m_calibrationFile = calibFile;

  std::vector<const CameraCalibration*> wanted;
  if (labels.empty()) {
    for (const auto& c : calib.cameras) wanted.push_back(&c);
  } else {
    for (const auto& l : labels) {
      const CameraCalibration* c = calib.find(l);
      if (!c) throw std::runtime_error("no EUCM camera " + l + " in " + calibFile.string());
      wanted.push_back(c);
    }
  }

  for (const CameraCalibration* c : wanted) {
    const auto files = findStreams(dir, c->deviceId);
    if (!files) {
      if (!labels.empty()) throw std::runtime_error("no streams of " + c->label + " (" + c->deviceId + ") in " + dir.string());
      spdlog::info("{} ({}) has no streams in the recording, skipped", c->label, c->deviceId);
      continue;
    }
    CameraStream stream(files->first, files->second);
    const CameraMeta& meta = stream.meta();
    if (meta.serial != c->deviceId)
      spdlog::warn("{}: stream serial {} differs from the device id {}", c->label, meta.serial, c->deviceId);
    if (meta.width != c->width || meta.height != c->height)
      throw std::runtime_error(c->label + ": stream " + std::to_string(meta.width) + "x" + std::to_string(meta.height) +
                               " does not match the calibration " + std::to_string(c->width) + "x" +
                               std::to_string(c->height));
    m_cameras.push_back(*c);
    m_streams.push_back(std::move(stream));
  }
  if (m_cameras.empty()) throw std::runtime_error("no calibrated camera streams in " + dir.string());

  for (int cam = 0; cam < size(); ++cam) {
    std::vector<std::uint64_t> ids;
    ids.reserve(m_streams[cam].size());
    // Frame id 0 means "no id"; aimio's camera group never delivers it.
    for (const auto& f : m_streams[cam].meta().frames)
      if (f.frameId != 0) ids.push_back(f.frameId);
    if (cam == 0) {
      m_synced = std::move(ids);
      continue;
    }
    std::vector<std::uint64_t> common;
    std::ranges::set_intersection(m_synced, ids, std::back_inserter(common));
    m_synced = std::move(common);
  }
  if (m_synced.empty()) spdlog::warn("the cameras of {} share no frame id", dir.string());
}

void Recording::selectFrames(std::vector<std::uint64_t> frameIds) {
  for (int cam = 0; cam < size(); ++cam) {
    const auto missing = std::ranges::count_if(frameIds, [&](std::uint64_t id) { return !m_streams[cam].meta().indexOf(id); });
    if (missing > 0)
      throw std::runtime_error(m_cameras[cam].label + " lacks " + std::to_string(missing) + " of the " +
                               std::to_string(frameIds.size()) + " selected frame ids");
  }
  m_synced = std::move(frameIds);
}

int Recording::indexOf(const std::string& label) const {
  for (int i = 0; i < size(); ++i)
    if (m_cameras[i].label == label) return i;
  return -1;
}

std::int64_t Recording::timestampNs(std::uint64_t frameId) const {
  // As aimio's camera group: the earliest valid exposure start of the group stands for all its images.
  std::int64_t earliest = 0;
  for (const auto& s : m_streams) {
    const auto index = s.meta().indexOf(frameId);
    if (!index) throw std::out_of_range("no frame id " + std::to_string(frameId));
    const std::int64_t t = s.meta().frames[*index].timestampNs;
    if (t > 0 && (earliest == 0 || t < earliest)) earliest = t;
  }
  return earliest;
}

std::vector<cv::Mat> Recording::read(std::uint64_t frameId, ImageFormat format) {
  std::vector<std::future<cv::Mat>> jobs;
  jobs.reserve(m_streams.size());
  for (auto& s : m_streams)
    jobs.push_back(std::async(std::launch::async, [&s, frameId, format] { return s.decodeFrameId(frameId, format); }));
  std::vector<cv::Mat> images;
  images.reserve(jobs.size());
  for (auto& j : jobs) images.push_back(j.get());
  for (int cam = 0; cam < size(); ++cam)
    if (images[cam].empty())
      throw std::out_of_range(m_cameras[cam].label + " has no frame id " + std::to_string(frameId));
  return images;
}

}  // namespace sdv::aimrec
