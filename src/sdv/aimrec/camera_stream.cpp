#include <sdv/aimrec/camera_stream.h>

#include <atomic>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#ifdef SDV_WITH_CUDA
#include <sdv/aimrec/gpu_image.h>
#endif

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/error.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
#ifdef SDV_WITH_CUDA
#include <libavutil/hwcontext_cuda.h>
#endif
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

std::atomic<DecodeDevice> g_device{DecodeDevice::Auto};

#ifdef SDV_WITH_CUDA
// One CUDA device for all streams, on the primary context so that the CUDA runtime of the converter shares it.
AVBufferRef* cudaDevice() {
  static AVBufferRef* device = nullptr;
  static std::once_flag once;
  std::call_once(once, [] {
    if (const int err = av_hwdevice_ctx_create(&device, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr,
                                               AV_CUDA_USE_PRIMARY_CONTEXT);
        err < 0) {
      spdlog::warn("no CUDA device for video decoding ({}), decoding on the CPU", avError(err));
      device = nullptr;
    }
  });
  return device;
}

AVPixelFormat chooseFormat(AVCodecContext*, const AVPixelFormat* formats) {
  for (const AVPixelFormat* f = formats; *f != AV_PIX_FMT_NONE; ++f)
    if (*f == AV_PIX_FMT_CUDA) return *f;
  return formats[0];
}
#endif

}  // namespace

void setDecodeDevice(DecodeDevice device) { g_device = device; }
DecodeDevice decodeDevice() { return g_device; }

DecodeDevice parseDecodeDevice(const std::string& name) {
  if (name == "cpu") return DecodeDevice::Cpu;
  if (name == "gpu") return DecodeDevice::Gpu;
  if (name == "auto") return DecodeDevice::Auto;
  throw std::invalid_argument("decode device must be cpu, gpu or auto, not " + name);
}

struct CameraStream::Decoder {
  std::ifstream file;
  std::string name;
  AVCodecContext* ctx = nullptr;
  AVPacket* packet = nullptr;
  AVFrame* frame = nullptr;  // the requested record
  AVFrame* received = nullptr;
  AVFrame* download = nullptr;
  SwsContext* swsGray = nullptr;
  SwsContext* swsBgr = nullptr;
  std::vector<std::int8_t> isIdr;  // -1 unknown
  std::optional<std::size_t> next;  // record the running decoder continues with
  std::optional<std::size_t> lastOutput;  // newest record the running decoder has output
  std::map<std::size_t, AVFrame*> ready;  // decoded records not returned yet
  bool drained = false;
  bool needsHeader = true;
  bool gpu = false;
  bool warnedFormat = false;
#ifdef SDV_WITH_CUDA
  std::unique_ptr<gpu::Converter> converter;
#endif

  ~Decoder() {
    clearReady();
    sws_freeContext(swsGray);
    sws_freeContext(swsBgr);
    av_frame_free(&frame);
    av_frame_free(&received);
    av_frame_free(&download);
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

  void clearReady() {
    for (auto& [index, f] : ready) av_frame_free(&f);
    ready.clear();
  }

  void reset() {
    avcodec_flush_buffers(ctx);
    clearReady();
    next.reset();
    lastOutput.reset();
    drained = false;
    needsHeader = true;
  }

  // Sends the next record (pts = record index) and collects the output frames from record `keep` on.
  void sendNext(const CameraMeta& meta, std::size_t keep) {
    const std::size_t index = *next;
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
    next = index + 1;
    if (err < 0) spdlog::warn("{}: record {} rejected by the decoder: {}", name, index, avError(err));
    collect(keep);
  }

  void drain(std::size_t keep) {
    avcodec_send_packet(ctx, nullptr);
    collect(keep);
    drained = true;
  }

  void collect(std::size_t keep) {
    for (;;) {
      const int err = avcodec_receive_frame(ctx, received);
      if (err == AVERROR(EAGAIN) || err == AVERROR_EOF) return;
      if (err < 0) throw std::runtime_error(name + ": decoding failed: " + avError(err));
      const auto index = static_cast<std::size_t>(received->pts);
      lastOutput = lastOutput ? std::max(*lastOutput, index) : index;
      if (index < keep || ready.contains(index)) {
        av_frame_unref(received);
        continue;
      }
      AVFrame* f = av_frame_alloc();
      if (!f) throw std::bad_alloc();
      av_frame_move_ref(f, received);
      ready.emplace(index, f);
    }
  }

  // The colour standard recorded in the meta header (or implied by the camera vendor) wins over the stream's VUI.
  static void colorMatrix(const AVFrame* f, AVPixelFormat src, ColorStandard standard, bool& bt601, bool& fullRange) {
    fullRange = standard == ColorStandard::Bt601Full || standard == ColorStandard::Bt709Full;
    bt601 = standard == ColorStandard::Bt601Full || standard == ColorStandard::Bt601Limited;
    if (standard == ColorStandard::Unknown) {
      fullRange = f->color_range == AVCOL_RANGE_JPEG || src == AV_PIX_FMT_YUVJ420P;
      bt601 = f->colorspace != AVCOL_SPC_BT709;
    }
  }

  cv::Mat convert(ImageFormat format, ColorStandard standard, std::span<const double> scales) {
    AVFrame* f = frame;
    if (frame->format == AV_PIX_FMT_CUDA) {
#ifdef SDV_WITH_CUDA
      const auto* frames = reinterpret_cast<const AVHWFramesContext*>(frame->hw_frames_ctx->data);
      if (frames->sw_format == AV_PIX_FMT_NV12) {
        bool bt601, fullRange;
        colorMatrix(frame, frames->sw_format, standard, bt601, fullRange);
        const int channels = format == ImageFormat::Gray ? 1 : 3;
        int w, h;
        gpu::scaledSize(frame->width, frame->height, scales, w, h);
        cv::Mat out(h, w, channels == 1 ? CV_8UC1 : CV_8UC3);
        const gpu::Nv12Frame nv12{frame->data[0], static_cast<std::size_t>(frame->linesize[0]), frame->data[1],
                                  static_cast<std::size_t>(frame->linesize[1]), frame->width, frame->height};
        converter->convert(nv12, {bt601, fullRange}, channels, scales, out.data, out.step);
        av_frame_unref(frame);
        return out;
      }
      if (!warnedFormat)
        spdlog::warn("{}: NVDEC delivers {}, converted on the CPU", name, av_get_pix_fmt_name(frames->sw_format));
      warnedFormat = true;
#endif
      if (const int err = av_hwframe_transfer_data(download, frame, 0); err < 0)
        throw std::runtime_error(name + ": cannot download the decoded frame: " + avError(err));
      av_frame_copy_props(download, frame);
      av_frame_unref(frame);
      f = download;
    }

    const int w = f->width, h = f->height;
    const bool gray = format == ImageFormat::Gray;
    SwsContext*& sws = gray ? swsGray : swsBgr;
    const AVPixelFormat src = static_cast<AVPixelFormat>(f->format);
    sws = sws_getCachedContext(sws, w, h, withoutJpegRange(src), w, h, gray ? AV_PIX_FMT_GRAY8 : AV_PIX_FMT_BGR24,
                               SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (!sws) throw std::runtime_error(name + ": unsupported decoded pixel format");
    bool bt601, fullRange;
    colorMatrix(f, src, standard, bt601, fullRange);
    sws_setColorspaceDetails(sws, sws_getCoefficients(bt601 ? SWS_CS_ITU601 : SWS_CS_ITU709), fullRange ? 1 : 0,
                             sws_getCoefficients(SWS_CS_DEFAULT), 1, 0, 1 << 16, 1 << 16);

    cv::Mat out(h, w, gray ? CV_8UC1 : CV_8UC3);
    std::uint8_t* dst[4] = {out.data, nullptr, nullptr, nullptr};
    const int dstStride[4] = {static_cast<int>(out.step), 0, 0, 0};
    sws_scale(sws, f->data, f->linesize, 0, h, dst, dstStride);
    av_frame_unref(f);
    for (const double scale : scales)
      if (scale != 1.0) cv::resize(out, out, {}, scale, scale, cv::INTER_AREA);
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

  AVBufferRef* device = nullptr;
#ifdef SDV_WITH_CUDA
  if (g_device != DecodeDevice::Cpu) {
    device = cudaDevice();
    if (!device && g_device == DecodeDevice::Gpu)
      throw std::runtime_error(d.name + ": GPU decoding requested, but there is no CUDA device");
  }
#else
  if (g_device == DecodeDevice::Gpu) throw std::runtime_error("GPU decoding requested, but built without CUDA");
#endif
  // NVDEC through NVIDIA's decoder wrapper (h264_cuvid), not FFmpeg's h264 hwaccel, which waits for every frame
  // (6 cameras: 51 instead of 10 frames/s).
  const AVCodec* cuvid = device ? avcodec_find_decoder_by_name("h264_cuvid") : nullptr;
  const AVCodec* codec = cuvid ? cuvid : avcodec_find_decoder(AV_CODEC_ID_H264);
  if (!codec) throw std::runtime_error("FFmpeg has no H.264 decoder");
  d.ctx = avcodec_alloc_context3(codec);
  d.packet = av_packet_alloc();
  d.frame = av_frame_alloc();
  d.received = av_frame_alloc();
  d.download = av_frame_alloc();
  if (!d.ctx || !d.packet || !d.frame || !d.received || !d.download) throw std::bad_alloc();
#ifdef SDV_WITH_CUDA
  if (device) {
    d.ctx->hw_device_ctx = av_buffer_ref(device);
    d.ctx->get_format = chooseFormat;
    d.converter = std::make_unique<gpu::Converter>();
    d.gpu = true;
  }
#endif
  // Low delay: without it the NVDEC wrapper outputs a frame only after four more were sent, which costs random
  // access more than it gains sequentially.
  d.ctx->flags |= AV_CODEC_FLAG_LOW_DELAY;
  if (!cuvid) {
    // Slice threads only: frame threading delays the output, which random access would have to drain.
    d.ctx->thread_count = decoderThreads;
    d.ctx->thread_type = FF_THREAD_SLICE;
  }
  if (const int err = avcodec_open2(d.ctx, codec, nullptr); err < 0)
    throw std::runtime_error(d.name + ": cannot open the H.264 decoder: " + avError(err));
}

CameraStream::~CameraStream() = default;
CameraStream::CameraStream(CameraStream&&) noexcept = default;
CameraStream& CameraStream::operator=(CameraStream&&) noexcept = default;

bool CameraStream::gpu() const { return m_decoder->gpu; }

cv::Mat CameraStream::decode(std::size_t index, ImageFormat format, std::span<const double> scales) {
  if (index >= size()) throw std::out_of_range(m_decoder->name + ": frame index " + std::to_string(index));
  Decoder& d = *m_decoder;

  std::erase_if(d.ready, [&](auto& r) {
    if (r.first >= index) return false;
    av_frame_free(&r.second);
    return true;
  });
  if (!d.ready.contains(index)) {
    std::size_t key = index;
    while (key > 0 && !d.idr(m_meta, key)) --key;
    if (key == 0 && !d.idr(m_meta, 0)) spdlog::warn("{}: no IDR frame before record {}", d.name, index);
    // Only going back needs a flush: forward, the running decoder skips to the IDR frame if that is closer.
    const bool back = d.drained || !d.next || (d.lastOutput && *d.lastOutput >= index);
    if (back) {
      d.reset();
      d.next = key;
    } else if (*d.next < key) {
      d.next = key;
    }
    // A record the decoder drops (damaged) must not make it decode the rest of the stream.
    constexpr std::size_t kMaxDelay = 16;
    while (!d.ready.contains(index) && *d.next < size() && *d.next <= index + kMaxDelay &&
           !(d.lastOutput && *d.lastOutput > index))
      d.sendNext(m_meta, index);
    if (!d.ready.contains(index) && !d.drained) d.drain(index);
  }
  const auto it = d.ready.find(index);
  if (it == d.ready.end()) {
    d.reset();
    throw std::runtime_error(d.name + ": record " + std::to_string(index) + " could not be decoded");
  }
  av_frame_move_ref(d.frame, it->second);
  av_frame_free(&it->second);
  d.ready.erase(it);
  return d.convert(format, m_meta.colorStandard, scales);
}

cv::Mat CameraStream::decodeFrameId(std::uint64_t frameId, ImageFormat format, std::span<const double> scales) {
  const auto index = m_meta.indexOf(frameId);
  return index ? decode(*index, format, scales) : cv::Mat();
}

}  // namespace sdv::aimrec
