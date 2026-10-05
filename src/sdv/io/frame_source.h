#pragma once

#include <array>
#include <filesystem>
#include <memory>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <sdv/io/kitti.h>

namespace sdv {

namespace aimrec {
class Recording;
}

// Sequential image input. Frames come as stored (colour or grey); next() returns an empty Mat at the end,
// skip() advances by one frame without decoding where possible and returns false at the end.
class FrameSource {
 public:
  virtual ~FrameSource() = default;
  virtual cv::Mat next() = 0;
  virtual bool skip() { return !next().empty(); }
  virtual void rewind() = 0;
  // Asks the source to resize its frames by `scale` (INTER_AREA, after any resizing of its own), replacing an earlier
  // request; false if it cannot, then the caller resizes.
  virtual bool setOutputScale(double /*scale*/) { return false; }
};

class VideoSource : public FrameSource {
 public:
  explicit VideoSource(const std::filesystem::path& file);
  cv::Mat next() override;
  bool skip() override { return m_capture.grab(); }
  void rewind() override;

 private:
  std::filesystem::path m_file;
  cv::VideoCapture m_capture;
};

// All images of a directory in lexicographic file-name order.
class ImageFolderSource : public FrameSource {
 public:
  explicit ImageFolderSource(const std::filesystem::path& dir);
  cv::Mat next() override;
  bool skip() override { return m_next < m_files.size() ? (++m_next, true) : false; }
  void rewind() override { m_next = 0; }

 private:
  std::vector<std::filesystem::path> m_files;
  size_t m_next = 0;
};

class KittiSource : public FrameSource {
 public:
  KittiSource(const KittiSequence& seq, int camIndex) : m_seq(seq), m_camIndex(camIndex) {}
  cv::Mat next() override { return m_next < m_seq.size() ? m_seq.loadImage(m_next++, m_camIndex) : cv::Mat(); }
  bool skip() override { return m_next < m_seq.size() ? (++m_next, true) : false; }
  void rewind() override { m_next = 0; }

 private:
  const KittiSequence& m_seq;
  int m_camIndex;
  size_t m_next = 0;
};

// Frames start, start + stride, start + 2 * stride, ... of another source.
class SubsampledSource : public FrameSource {
 public:
  SubsampledSource(std::unique_ptr<FrameSource> source, size_t start, size_t stride);
  cv::Mat next() override;
  bool skip() override;
  void rewind() override;
  bool setOutputScale(double scale) override { return m_source->setOutputScale(scale); }

 private:
  std::unique_ptr<FrameSource> m_source;
  size_t m_start, m_stride;
  bool m_started = false;
};

// Frames of another source resized by `scale` (INTER_AREA).
class ScaledSource : public FrameSource {
 public:
  ScaledSource(std::unique_ptr<FrameSource> source, double scale) : m_source(std::move(source)), m_scale(scale) {}
  cv::Mat next() override;
  bool skip() override { return m_source->skip(); }
  void rewind() override { m_source->rewind(); }

 private:
  std::unique_ptr<FrameSource> m_source;
  double m_scale;
};

// One camera of an aiMotive recording (BGR), over the frame ids that all open cameras of the recording have, so the
// sources of one recording stay synchronised when a camera drops a frame. Skipping does not decode. `imageScale`:
// INTER_AREA resize of every frame; it and the output scale are applied by the decoder (on the GPU where it decodes).
class AimRecordSource : public FrameSource {
 public:
  AimRecordSource(std::shared_ptr<aimrec::Recording> recording, int camera, double imageScale = 1.0);
  cv::Mat next() override;
  bool skip() override;
  void rewind() override { m_next = 0; }
  bool setOutputScale(double scale) override;

 private:
  std::shared_ptr<aimrec::Recording> m_recording;
  int m_camera;
  std::array<double, 2> m_scales;  // image scale, output scale
  size_t m_next = 0;
};

}  // namespace sdv
