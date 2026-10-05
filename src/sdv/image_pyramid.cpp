#include <sdv/image_pyramid.h>

#include <stdexcept>

#include <opencv2/imgproc.hpp>

#include <sdv/parallel.h>

namespace sdv {

namespace {

template <typename RowBody>
void parallelRows(int rows, RowBody&& rowBody) {
  parallelChunks(static_cast<size_t>(rows), [&](size_t, size_t begin, size_t end) {
    for (size_t v = begin; v < end; ++v) rowBody(static_cast<int>(v));
  });
}

void fillGradients(ImageLevel& lvl) {
  const int w = lvl.width, h = lvl.height;
  lvl.gradNormSq.assign(static_cast<size_t>(w) * h, 0.f);
  lvl.intensity.resize(static_cast<size_t>(w) * h);
  for (size_t i = 0; i < lvl.intensity.size(); ++i) lvl.intensity[i] = lvl.data[i][0];
  // Central differences; border pixels keep zero gradient.
  parallelRows(h, [&](int v) {
    if (v == 0 || v == h - 1) return;
    for (int u = 1; u < w - 1; ++u) {
      const int i = v * w + u;
      const float gu = 0.5f * (lvl.data[i + 1][0] - lvl.data[i - 1][0]);
      const float gv = 0.5f * (lvl.data[i + w][0] - lvl.data[i - w][0]);
      lvl.data[i][1] = gu;
      lvl.data[i][2] = gv;
      lvl.gradNormSq[i] = gu * gu + gv * gv;
    }
  });
}

}  // namespace

cv::Mat toFloatGray(const cv::Mat& image) {
  cv::Mat gray = image;
  if (image.channels() == 3) cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  cv::Mat out;
  const double scale = gray.depth() == CV_16U ? 255.0 / 65535.0 : 1.0;
  gray.convertTo(out, CV_32F, scale);
  return out;
}

ImagePyramid::ImagePyramid(const cv::Mat& image, int levels) {
  if (image.type() != CV_32FC1) throw std::invalid_argument("ImagePyramid expects CV_32FC1");
  const int div = 1 << (levels - 1);
  if (image.cols % div != 0 || image.rows % div != 0)
    throw std::invalid_argument("image size must be divisible by 2^(levels-1)");

  m_levels.resize(levels);
  ImageLevel& base = m_levels[0];
  base.width = image.cols;
  base.height = image.rows;
  base.data.resize(static_cast<size_t>(base.width) * base.height);
  parallelRows(base.height, [&](int v) {
    const float* row = image.ptr<float>(v);
    for (int u = 0; u < base.width; ++u) base.data[v * base.width + u] = {row[u], 0.f, 0.f};
  });
  fillGradients(base);

  for (int l = 1; l < levels; ++l) {
    const ImageLevel& fine = m_levels[l - 1];
    ImageLevel& lvl = m_levels[l];
    lvl.width = fine.width / 2;
    lvl.height = fine.height / 2;
    lvl.data.resize(static_cast<size_t>(lvl.width) * lvl.height);
    parallelRows(lvl.height, [&](int v) {
      for (int u = 0; u < lvl.width; ++u) {
        const int f = 2 * v * fine.width + 2 * u;
        const float i = 0.25f * (fine.data[f][0] + fine.data[f + 1][0] + fine.data[f + fine.width][0] +
                                 fine.data[f + fine.width + 1][0]);
        lvl.data[v * lvl.width + u] = {i, 0.f, 0.f};
      }
    });
    fillGradients(lvl);
  }
}

}  // namespace sdv
