#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace sdv::aimrec::gpu {

// NV12 frame in device memory: full-resolution luma plane and interleaved half-resolution chroma (U, V).
struct Nv12Frame {
  const std::uint8_t* y;
  std::size_t yPitch;
  const std::uint8_t* uv;
  std::size_t uvPitch;
  int width, height;
};

struct ColorMatrix {
  bool bt601 = true;
  bool fullRange = false;
};

// Output size of successive INTER_AREA resizes by these factors (cv::resize with fx = fy = scale).
void scaledSize(int width, int height, std::span<const double> scales, int& outWidth, int& outHeight);

// NV12 to 8-bit grey (luma) or BGR, then area-averaged downscales as cv::resize(INTER_AREA) with each factor in
// turn, each rounded to 8 bit like the CPU chain; only the result is copied to the host. One converter per thread.
class Converter {
 public:
  Converter();
  ~Converter();
  Converter(const Converter&) = delete;
  Converter& operator=(const Converter&) = delete;

  // `dst`: host memory of the scaledSize() image with `channels` (1 or 3) and row step `dstStep`.
  void convert(const Nv12Frame& src, ColorMatrix color, int channels, std::span<const double> scales,
               std::uint8_t* dst, std::size_t dstStep);

 private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
};

}  // namespace sdv::aimrec::gpu
