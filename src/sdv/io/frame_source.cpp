#include <sdv/io/frame_source.h>

#include <algorithm>
#include <cctype>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <sdv/aimrec/recording.h>

namespace sdv {

VideoSource::VideoSource(const std::filesystem::path& file) : m_file(file) { rewind(); }

void VideoSource::rewind() {
  m_capture.open(m_file.string(), cv::CAP_FFMPEG);
  if (!m_capture.isOpened()) throw std::runtime_error("cannot open video " + m_file.string());
}

cv::Mat VideoSource::next() {
  cv::Mat frame;
  if (!m_capture.read(frame)) return {};
  return frame;
}

ImageFolderSource::ImageFolderSource(const std::filesystem::path& dir) {
  for (const auto& e : std::filesystem::directory_iterator(dir)) {
    std::string ext = e.path().extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (e.is_regular_file() && (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".tif" ||
                                ext == ".tiff"))
      m_files.push_back(e.path());
  }
  if (m_files.empty()) throw std::runtime_error("no images in " + dir.string());
  std::sort(m_files.begin(), m_files.end());
}

cv::Mat ImageFolderSource::next() {
  if (m_next >= m_files.size()) return {};
  cv::Mat img = cv::imread(m_files[m_next].string(), cv::IMREAD_UNCHANGED);
  if (img.empty()) throw std::runtime_error("cannot read " + m_files[m_next].string());
  ++m_next;
  return img;
}

SubsampledSource::SubsampledSource(std::unique_ptr<FrameSource> source, size_t start, size_t stride)
    : m_source(std::move(source)), m_start(start), m_stride(std::max<size_t>(stride, 1)) {}

cv::Mat SubsampledSource::next() {
  size_t skip = m_started ? m_stride - 1 : m_start;
  m_started = true;
  for (; skip > 0; --skip)
    if (!m_source->skip()) return {};
  return m_source->next();
}

bool SubsampledSource::skip() {
  size_t n = m_started ? m_stride : m_start + 1;
  m_started = true;
  for (; n > 0; --n)
    if (!m_source->skip()) return false;
  return true;
}

cv::Mat ScaledSource::next() {
  cv::Mat image = m_source->next();
  if (image.empty() || m_scale == 1.0) return image;
  cv::Mat out;
  cv::resize(image, out, {}, m_scale, m_scale, cv::INTER_AREA);
  return out;
}

AimRecordSource::AimRecordSource(std::shared_ptr<aimrec::Recording> recording, int camera)
    : m_recording(std::move(recording)), m_camera(camera) {}

cv::Mat AimRecordSource::next() {
  const auto& ids = m_recording->syncedFrameIds();
  if (m_next >= ids.size()) return {};
  return m_recording->stream(m_camera).decodeFrameId(ids[m_next++], aimrec::ImageFormat::Bgr);
}

bool AimRecordSource::skip() {
  if (m_next >= m_recording->syncedFrameIds().size()) return false;
  ++m_next;
  return true;
}

void SubsampledSource::rewind() {
  m_source->rewind();
  m_started = false;
}

}  // namespace sdv
