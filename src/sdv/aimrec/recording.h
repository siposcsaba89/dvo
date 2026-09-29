#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include <sdv/aimrec/calibration.h>
#include <sdv/aimrec/camera_stream.h>

namespace sdv::aimrec {

// Cameras of an aiMotive recording directory (Camera_<vendor>_<nn>_<device id>_{meta.dat,data.h264}, 00_session.yml)
// with their calibration. The cameras are hardware triggered: frames of one trigger share the frame id, which is
// the synchronisation key.
class Recording {
 public:
  // `calibration`: aimprototype_config.zip or a sensorconfig.yaml; empty = aimprototype_config.zip next to the
  // record directory. `labels`: cameras to open in this order; empty = every calibrated camera that has streams.
  // Without `loadMasks` the calibrations come without obstruction masks (the caller has its own).
  explicit Recording(const std::filesystem::path& recordDir, const std::filesystem::path& calibration = {},
                     const std::vector<std::string>& labels = {}, bool loadMasks = true);

  const std::string& vehicle() const { return m_vehicle; }
  const std::filesystem::path& recordDir() const { return m_recordDir; }
  const std::filesystem::path& calibrationFile() const { return m_calibrationFile; }
  int size() const { return static_cast<int>(m_cameras.size()); }
  // -1 if the camera is not open.
  int indexOf(const std::string& label) const;
  const CameraCalibration& calibration(int cam) const { return m_cameras[cam]; }
  CameraStream& stream(int cam) { return m_streams[cam]; }

  // Frame ids recorded by every open camera, increasing. Only the open cameras count, so a subset of cameras can
  // have more synced frames than the whole rig.
  const std::vector<std::uint64_t>& syncedFrameIds() const { return m_synced; }
  // Replaces the synced frame ids, e.g. by those of another camera selection so that both run over the same
  // frames. Throws if an open camera lacks one of them.
  void selectFrames(std::vector<std::uint64_t> frameIds);
  // Earliest exposure start (host clock, ns) over the open cameras for this frame id; per-camera times are in
  // stream(cam).meta().
  std::int64_t timestampNs(std::uint64_t frameId) const;
  // One image per open camera, decoded in parallel. Throws if a camera lacks the frame.
  std::vector<cv::Mat> read(std::uint64_t frameId, ImageFormat format = ImageFormat::Gray);

 private:
  std::string m_vehicle;
  std::filesystem::path m_recordDir, m_calibrationFile;
  std::vector<CameraCalibration> m_cameras;
  std::vector<CameraStream> m_streams;
  std::vector<std::uint64_t> m_synced;
};

}  // namespace sdv::aimrec
