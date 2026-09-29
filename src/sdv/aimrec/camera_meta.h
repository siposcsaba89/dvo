#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace sdv::aimrec {

enum class ColorStandard : std::uint8_t { Unknown = 0, Bt601Full = 2, Bt601Limited = 3, Bt709Full = 4, Bt709Limited = 5 };

struct FrameMeta {
  std::uint64_t frameId = 0;      // trigger counter, shared by the synchronised cameras of a recording
  std::int64_t timestampNs = 0;   // synchronised host clock at exposure start
  std::int64_t receivedNs = 0;    // host clock when the frame arrived
  std::uint64_t deviceTimestamp = 0;
  std::uint64_t offset = 0;       // record position in the data file
  std::uint32_t size = 0;         // record size in bytes
  float gain = 0.0f;              // dB, NaN if unknown
  float exposureUs = 0.0f;        // NaN if unknown
};

// *_meta.dat of a camera: record file header, camera header, one fixed-size record per frame in the data file.
struct CameraMeta {
  int version = 0;
  std::string serial;
  int width = 0, height = 0;
  float fps = 0.0f;
  bool h264 = false;
  std::uint32_t dataHeaderSize = 0;  // bytes at the start of the data file preceding the first record (SPS/PPS)
  int gopLength = 0;                 // 0 if unknown
  ColorStandard colorStandard = ColorStandard::Unknown;
  std::vector<FrameMeta> frames;     // in recording order, frame ids strictly increasing (checked on parse)

  std::optional<std::size_t> indexOf(std::uint64_t frameId) const;
  // Frame with the closest timestamp.
  std::size_t nearestIndex(std::int64_t timestampNs) const;
};

CameraMeta readCameraMeta(const std::filesystem::path& file);
CameraMeta parseCameraMeta(const std::vector<std::uint8_t>& bytes, const std::string& name);

}  // namespace sdv::aimrec
