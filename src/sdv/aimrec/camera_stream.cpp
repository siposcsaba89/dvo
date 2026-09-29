#include <sdv/aimrec/camera_stream.h>

#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/error.h>
#include <libswscale/swscale.h>
}

namespace sdv::aimrec {

namespace {

std::string avError(int err) {
  char buf[AV_ERROR_MAX_STRING_SIZE] = {};
  av_strerror(err, buf, sizeof(buf));
  return buf;
}

// First VCL NAL unit type of an Annex B access unit (5 = IDR slice), or -1 if none is in `bytes`.
int firstSliceType(const std::uint8_t* bytes, std::size_t n) {
  for (std::size_t i = 0; i + 3 < n; ++i) {
    if (bytes[i] != 0 || bytes[i + 1] != 0 || bytes[i + 2] != 1) continue;
    const int type = bytes[i + 3] & 0x1f;
    if (type >= 1 && type <= 5) return type;
    i += 2;
  }
  return -1;
}

// Deprecated full-range aliases; the range is set explicitly instead.
AVPixelFormat withoutJpegRange(AVPixelFormat f) {
  switch (f) {
    case AV_PIX_FMT_YUVJ420P: return AV_PIX_FMT_YUV420P;
    case AV_PIX_FMT_YUVJ422P: return AV_PIX_FMT_YUV422P;
    case AV_PIX_FMT_YUVJ444P: return AV_PIX_FMT_YUV444P;
    default: return f;
  }
}

}  // namespace

struct CameraStream::Decoder {
  std::ifstream file;
  std::string name;
  AVCodecContext* ctx = nullptr;
  AVPacket* packet = nullptr;
  AVFrame* frame = nullptr;
  SwsContext* swsGray = nullptr;
  SwsContext* swsBgr = nullptr;
  std::vector<std::int8_t> isIdr;  // -1 unknown
  std::optional<std::size_t> next;  // index the running decoder continues with
  bool needsHeader = true;

  ~Decoder() {
    sws_freeContext(swsGray);
    sws_freeContext(swsBgr);
    av_frame_free(&frame);
    av_packet_free(&packet);
    avcodec_free_context(&ctx);
  }

  void readRecord(const CameraMeta& meta, std::size_t index, std::uint32_t bytes, std::uint8_t* dst) {
    file.clear();
    file.seekg(static_cast<std::streamoff>(meta.frames[index].offset));
    file.read(reinterpret_cast<char*>(dst), bytes);
    if (!file) throw std::runtime_error(name + ": cannot read record " + std::to_string(index));
  }

  bool idr(const CameraMeta& meta, std::size_t index) {
    if (isIdr[index] < 0) {
      const FrameMeta& f = meta.frames[index];
      std::vector<std::uint8_t> head(std::min<std::uint32_t>(f.size, 4096));
      readRecord(meta, index, static_cast<std::uint32_t>(head.size()), head.data());
      int type = firstSliceType(head.data(), head.size());
      if (type < 0 && f.size > head.size()) {
        head.resize(f.size);
        readRecord(meta, index, f.size, head.data());
        type = firstSliceType(head.data(), head.size());
      }
      isIdr[index] = type == 5 ? 1 : 0;
    }
    return isIdr[index] == 1;
  }

  void reset() {
    avcodec_flush_buffers(ctx);
    next.reset();
    needsHeader = true;
  }

  // Sends record `index` (pts = record index); `got` is set once the frame of record `target` is in `frame`.
  void send(const CameraMeta& meta, std::size_t index, std::size_t target, bool& got) {
    const FrameMeta& f = meta.frames[index];
    const std::uint32_t header = needsHeader ? meta.dataHeaderSize : 0;
    if (av_new_packet(packet, static_cast<int>(header + f.size)) < 0) throw std::bad_alloc();
    if (header > 0) {
      file.clear();
      file.seekg(0);
      file.read(reinterpret_cast<char*>(packet->data), header);
      if (!file) throw std::runtime_error(name + ": cannot read the stream header");
    }
    readRecord(meta, index, f.size, packet->data + header);
    packet->pts = packet->dts = static_cast<std::int64_t>(index);
    const int err = avcodec_send_packet(ctx, packet);
    av_packet_unref(packet);
    needsHeader = false;
    if (err < 0) spdlog::warn("{}: record {} rejected by the decoder: {}", name, index, avError(err));
    receive(target, got);
  }

  void receive(std::size_t target, bool& got) {
    for (;;) {
      if (got) return;
      const int err = avcodec_receive_frame(ctx, frame);
      if (err == AVERROR(EAGAIN) || err == AVERROR_EOF) return;
      if (err < 0) throw std::runtime_error(name + ": decoding failed: " + avError(err));
      if (frame->pts == static_cast<std::int64_t>(target))
        got = true;
      else
        av_frame_unref(frame);
    }
  }

  cv::Mat convert(ImageFormat format, ColorStandard standard) {
    const int w = frame->width, h = frame->height;
    const bool gray = format == ImageFormat::Gray;
    SwsContext*& sws = gray ? swsGray : swsBgr;
    const AVPixelFormat src = static_cast<AVPixelFormat>(frame->format);
    sws = sws_getCachedContext(sws, w, h, withoutJpegRange(src), w, h, gray ? AV_PIX_FMT_GRAY8 : AV_PIX_FMT_BGR24,
                               SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (!sws) throw std::runtime_error(name + ": unsupported decoded pixel format");

    // The colour standard recorded in the meta header (or implied by the camera vendor) wins over the stream's VUI.
    bool fullRange = standard == ColorStandard::Bt601Full || standard == ColorStandard::Bt709Full;
    bool bt601 = standard == ColorStandard::Bt601Full || standard == ColorStandard::Bt601Limited;
    if (standard == ColorStandard::Unknown) {
      fullRange = frame->color_range == AVCOL_RANGE_JPEG || src == AV_PIX_FMT_YUVJ420P;
      bt601 = frame->colorspace != AVCOL_SPC_BT709;
    }
    sws_setColorspaceDetails(sws, sws_getCoefficients(bt601 ? SWS_CS_ITU601 : SWS_CS_ITU709), fullRange ? 1 : 0,
                             sws_getCoefficients(SWS_CS_DEFAULT), 1, 0, 1 << 16, 1 << 16);

    cv::Mat out(h, w, gray ? CV_8UC1 : CV_8UC3);
    std::uint8_t* dst[4] = {out.data, nullptr, nullptr, nullptr};
    const int dstStride[4] = {static_cast<int>(out.step), 0, 0, 0};
    sws_scale(sws, frame->data, frame->linesize, 0, h, dst, dstStride);
    av_frame_unref(frame);
    return out;
  }
};

CameraStream::CameraStream(const std::filesystem::path& metaFile, const std::filesystem::path& dataFile,
                           int decoderThreads)
    : m_meta(readCameraMeta(metaFile)), m_dataFile(dataFile), m_decoder(std::make_unique<Decoder>()) {
  Decoder& d = *m_decoder;
  d.name = dataFile.filename().string();
  if (!m_meta.h264) throw std::runtime_error(d.name + ": only H.264 camera streams are supported");
  d.file.open(dataFile, std::ios::binary);
  if (!d.file) throw std::runtime_error("cannot open " + dataFile.string());
  d.isIdr.assign(m_meta.frames.size(), -1);

  const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_H264);
  if (!codec) throw std::runtime_error("FFmpeg has no H.264 decoder");
  d.ctx = avcodec_alloc_context3(codec);
  d.packet = av_packet_alloc();
  d.frame = av_frame_alloc();
  if (!d.ctx || !d.packet || !d.frame) throw std::bad_alloc();
  // Slice threads only: frame threading delays the output, which random access would have to drain.
  d.ctx->thread_count = decoderThreads;
  d.ctx->thread_type = FF_THREAD_SLICE;
  d.ctx->flags |= AV_CODEC_FLAG_LOW_DELAY;
  if (const int err = avcodec_open2(d.ctx, codec, nullptr); err < 0)
    throw std::runtime_error(d.name + ": cannot open the H.264 decoder: " + avError(err));
}

CameraStream::~CameraStream() = default;
CameraStream::CameraStream(CameraStream&&) noexcept = default;
CameraStream& CameraStream::operator=(CameraStream&&) noexcept = default;

cv::Mat CameraStream::decode(std::size_t index, ImageFormat format) {
  if (index >= size()) throw std::out_of_range(m_decoder->name + ": frame index " + std::to_string(index));
  Decoder& d = *m_decoder;

  std::size_t key = index;
  while (key > 0 && !d.idr(m_meta, key)) --key;
  if (key == 0 && !d.idr(m_meta, 0)) spdlog::warn("{}: no IDR frame before record {}", d.name, index);

  std::size_t start = key;
  if (d.next && *d.next > key && *d.next <= index)
    start = *d.next;
  else
    d.reset();

  bool got = false;
  for (std::size_t i = start; i <= index; ++i) d.send(m_meta, i, index, got);
  d.next = index + 1;
  if (!got) {
    // Decoder holds the frame back (reordering); drain it.
    avcodec_send_packet(d.ctx, nullptr);
    d.receive(index, got);
    d.reset();
  }
  if (!got) throw std::runtime_error(d.name + ": record " + std::to_string(index) + " could not be decoded");
  return d.convert(format, m_meta.colorStandard);
}

cv::Mat CameraStream::decodeFrameId(std::uint64_t frameId, ImageFormat format) {
  const auto index = m_meta.indexOf(frameId);
  return index ? decode(*index, format) : cv::Mat();
}

}  // namespace sdv::aimrec
