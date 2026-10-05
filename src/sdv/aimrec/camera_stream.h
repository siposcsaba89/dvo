#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>
#include <string>

#include <opencv2/core.hpp>

#include <sdv/aimrec/camera_meta.h>

namespace sdv::aimrec {

enum class ImageFormat { Gray, Bgr };

// Cpu: FFmpeg's H.264 decoder. Gpu: NVDEC, with colour conversion and resizing in CUDA, so only the final image is
// copied to the host. Auto: Gpu where available, Cpu otherwise.
enum class DecodeDevice { Cpu, Gpu, Auto };
// For streams opened afterwards (process-wide).
void setDecodeDevice(DecodeDevice device);
DecodeDevice decodeDevice();
// "cpu", "gpu" or "auto".
DecodeDevice parseDecodeDevice(const std::string& name);

// Random access to the frames of one camera: *_meta.dat plus the H.264 elementary stream *_data.h264. Frames are
// decoded from the closest preceding IDR frame, sequential reads continue the running decoder. Not thread-safe;
// separate streams can be decoded in parallel.
class CameraStream {
 public:
  CameraStream(const std::filesystem::path& metaFile, const std::filesystem::path& dataFile, int decoderThreads = 1);
  ~CameraStream();
  CameraStream(CameraStream&&) noexcept;
  CameraStream& operator=(CameraStream&&) noexcept;

  const CameraMeta& meta() const { return m_meta; }
  const std::filesystem::path& dataFile() const { return m_dataFile; }
  std::size_t size() const { return m_meta.frames.size(); }

  // Gray: CV_8UC1 luma, Bgr: CV_8UC3; both full range. `scales`: successive resizes as cv::resize(INTER_AREA) with
  // fx = fy = scale (on the GPU when decoding there).
  cv::Mat decode(std::size_t index, ImageFormat format = ImageFormat::Gray, std::span<const double> scales = {});
  // Empty Mat if the camera has no frame with this id.
  cv::Mat decodeFrameId(std::uint64_t frameId, ImageFormat format = ImageFormat::Gray,
                        std::span<const double> scales = {});
  // Whether frames are decoded by NVDEC.
  bool gpu() const;

 private:
  struct Decoder;
  CameraMeta m_meta;
  std::filesystem::path m_dataFile;
  std::unique_ptr<Decoder> m_decoder;
};

}  // namespace sdv::aimrec
