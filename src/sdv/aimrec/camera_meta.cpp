#include <sdv/aimrec/camera_meta.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace sdv::aimrec {

namespace {

// Record file: 15-byte header (header size, version, metadata size, fixed item size, lookup flag, lookup size,
// compression), the metadata block (camera header) and the items (frame records). Little endian, packed.
constexpr std::size_t kFileHeaderSize = 15;
constexpr std::size_t kCameraHeaderV8Size = 222;
constexpr std::size_t kFrameRecordV8Size = 77;
constexpr std::size_t kFrameRecordV9Size = 81;

constexpr std::uint16_t kVendorAimotive = 0x0004;
constexpr std::uint16_t kVendorNvMedia = 0x8001;
constexpr std::uint16_t kVendorQCarCam = 0x8017;
constexpr std::uint16_t kVendorV4l2 = 0x8018;

template <typename T>
T get(const std::uint8_t* p) {
  T v;
  std::memcpy(&v, p, sizeof(T));
  return v;
}

}  // namespace

std::optional<std::size_t> CameraMeta::indexOf(std::uint64_t frameId) const {
  const auto it = std::lower_bound(frames.begin(), frames.end(), frameId,
                                   [](const FrameMeta& f, std::uint64_t id) { return f.frameId < id; });
  if (it != frames.end() && it->frameId == frameId) return static_cast<std::size_t>(it - frames.begin());
  return std::nullopt;
}

std::size_t CameraMeta::nearestIndex(std::int64_t timestampNs) const {
  if (frames.empty()) throw std::runtime_error("camera " + serial + " has no frames");
  const auto it = std::lower_bound(frames.begin(), frames.end(), timestampNs,
                                   [](const FrameMeta& f, std::int64_t t) { return f.timestampNs < t; });
  if (it == frames.begin()) return 0;
  if (it == frames.end()) return frames.size() - 1;
  const auto prev = it - 1;
  return static_cast<std::size_t>((timestampNs - prev->timestampNs <= it->timestampNs - timestampNs ? prev : it) -
                                  frames.begin());
}

CameraMeta parseCameraMeta(const std::vector<std::uint8_t>& bytes, const std::string& name) {
  const auto fail = [&](const std::string& what) { return std::runtime_error(name + ": " + what); };
  if (bytes.size() < kFileHeaderSize) throw fail("too short for a record file header");
  const std::uint8_t* b = bytes.data();
  const auto headerSize = get<std::uint16_t>(b);
  const auto headerVersion = get<std::uint16_t>(b + 2);
  const auto metadataSize = get<std::uint32_t>(b + 4);
  const auto itemSize = get<std::uint32_t>(b + 8);
  const auto useLookup = b[12];
  const auto compression = b[14];
  if (headerSize != kFileHeaderSize || headerVersion != 1) throw fail("not a record file");
  if (useLookup != 0 || compression != 0 || itemSize == 0)
    throw fail("lookup tables, compression or variable-size records are not supported");
  if (metadataSize < kCameraHeaderV8Size || bytes.size() < kFileHeaderSize + metadataSize)
    throw fail("camera header version older than 8 is not supported");

  const std::uint8_t* h = b + kFileHeaderSize;
  CameraMeta meta;
  meta.version = static_cast<int>(get<std::uint32_t>(h));
  if (meta.version < 8 || meta.version > 9) throw fail("unsupported camera meta version " + std::to_string(meta.version));
  const std::size_t expectedItem = meta.version == 8 ? kFrameRecordV8Size : kFrameRecordV9Size;
  if (itemSize != expectedItem) throw fail("unexpected frame record size " + std::to_string(itemSize));

  meta.h264 = h[5] == 1;
  meta.dataHeaderSize = get<std::uint32_t>(h + 6);
  const auto vendor = get<std::uint16_t>(h + 14);
  // A trigger box id below -1 encodes the GOP length of the H.264 stream.
  const auto triggerBoxId = get<std::int32_t>(h + 22);
  meta.gopLength = triggerBoxId < -1 ? -triggerBoxId : 0;
  meta.fps = get<float>(h + 26);
  meta.width = get<std::int32_t>(h + 86);
  meta.height = get<std::int32_t>(h + 90);
  const char* serial = reinterpret_cast<const char*>(h + 94);
  meta.serial.assign(serial, strnlen(serial, 128));
  if (meta.version >= 9 && metadataSize > kCameraHeaderV8Size)
    meta.colorStandard = static_cast<ColorStandard>(h[kCameraHeaderV8Size]);
  if (meta.colorStandard == ColorStandard::Unknown) {
    if (vendor == kVendorNvMedia || vendor == kVendorV4l2 || vendor == kVendorAimotive)
      meta.colorStandard = ColorStandard::Bt709Full;
    else if (vendor == kVendorQCarCam)
      meta.colorStandard = ColorStandard::Bt601Full;
  }

  // A partially written last record is ignored.
  const std::size_t first = kFileHeaderSize + metadataSize;
  const std::size_t count = (bytes.size() - first) / itemSize;
  meta.frames.resize(count);
  for (std::size_t i = 0; i < count; ++i) {
    const std::uint8_t* r = b + first + i * itemSize;
    FrameMeta& f = meta.frames[i];
    f.offset = get<std::uint64_t>(r);
    f.size = get<std::uint32_t>(r + 8);
    f.frameId = get<std::uint64_t>(r + 12);
    f.timestampNs = get<std::int64_t>(r + 20);
    f.receivedNs = get<std::int64_t>(r + 28);
    f.deviceTimestamp = get<std::uint64_t>(r + 36);
    f.gain = get<float>(r + 68);
    f.exposureUs = get<float>(r + 72);
    if (i > 0 && f.frameId <= meta.frames[i - 1].frameId)
      throw fail("frame ids do not increase at record " + std::to_string(i));
  }
  return meta;
}

CameraMeta readCameraMeta(const std::filesystem::path& file) {
  std::ifstream in(file, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + file.string());
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(std::filesystem::file_size(file)));
  in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (!in) throw std::runtime_error("cannot read " + file.string());
  return parseCameraMeta(bytes, file.string());
}

}  // namespace sdv::aimrec
