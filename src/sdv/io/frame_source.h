#pragma once

#include <filesystem>
#include <memory>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <sdv/io/kitti.h>

namespace sdv {

// Sequential image input. Frames come as stored (colour or grey); next() returns an empty Mat at the end,
// skip() advances by one frame without decoding where possible and returns false at the end.
class FrameSource {
 public:
  virtual ~FrameSource() = default;
  virtual cv::Mat next() = 0;
  virtual bool skip() { return !next().empty(); }
  virtual void rewind() = 0;
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
  void rewind() override;

 private:
  std::unique_ptr<FrameSource> m_source;
  size_t m_start, m_stride;
  bool m_started = false;
};

}  // namespace sdv
