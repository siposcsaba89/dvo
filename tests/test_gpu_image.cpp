#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

#include <sdv/aimrec/gpu_image.h>

namespace {

using namespace sdv::aimrec;

struct DeviceNv12 {
  std::uint8_t* y = nullptr;
  std::uint8_t* uv = nullptr;
  ~DeviceNv12() {
    cudaFree(y);
    cudaFree(uv);
  }
};

bool haveDevice() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

// Smooth random content, so that the area averages see gradients rather than noise only.
cv::Mat randomPlane(int w, int h) {
  cv::Mat small(h / 8 + 1, w / 8 + 1, CV_8UC1), plane, noise(h, w, CV_8UC1);
  cv::randu(small, 0, 256);
  cv::resize(small, plane, {w, h}, 0, 0, cv::INTER_CUBIC);
  cv::randu(noise, 0, 24);
  cv::add(plane, noise, plane);
  return plane;
}

cv::Mat gpuConvert(const cv::Mat& y, const cv::Mat& uv, gpu::ColorMatrix color, int channels,
                   const std::vector<double>& scales) {
  DeviceNv12 d;
  EXPECT_EQ(cudaMalloc(&d.y, y.total()), cudaSuccess);
  EXPECT_EQ(cudaMalloc(&d.uv, uv.total() * uv.elemSize()), cudaSuccess);
  cudaMemcpy(d.y, y.data, y.total(), cudaMemcpyHostToDevice);
  cudaMemcpy(d.uv, uv.data, uv.total() * uv.elemSize(), cudaMemcpyHostToDevice);
  int w, h;
  gpu::scaledSize(y.cols, y.rows, scales, w, h);
  cv::Mat out(h, w, channels == 1 ? CV_8UC1 : CV_8UC3);
  gpu::Converter converter;
  converter.convert({d.y, static_cast<std::size_t>(y.cols), d.uv, static_cast<std::size_t>(uv.cols) * 2, y.cols, y.rows},
                    color, channels, scales, out.data, out.step);
  return out;
}

TEST(GpuImage, ColourConversionMatchesTheStandard) {
  if (!haveDevice()) GTEST_SKIP() << "no CUDA device";
  const int w = 96, h = 64;
  const cv::Mat y = randomPlane(w, h);
  cv::Mat uv(h / 2, w / 2, CV_8UC2);
  cv::randu(uv, 0, 256);
  for (const bool bt601 : {true, false})
    for (const bool full : {true, false}) {
      const cv::Mat bgr = gpuConvert(y, uv, {bt601, full}, 3, {});
      const cv::Mat gray = gpuConvert(y, uv, {bt601, full}, 1, {});
      const double kr = bt601 ? 0.299 : 0.2126, kb = bt601 ? 0.114 : 0.0722, kg = 1 - kr - kb;
      double maxError = 0;
      for (int r = 0; r < h; ++r)
        for (int c = 0; c < w; ++c) {
          const double luma = full ? y.at<std::uint8_t>(r, c) : (y.at<std::uint8_t>(r, c) - 16) * 255.0 / 219.0;
          const cv::Vec2b ch = uv.at<cv::Vec2b>(r / 2, c / 2);
          const double cs = full ? 1.0 : 255.0 / 224.0, u = cs * (ch[0] - 128.0), v = cs * (ch[1] - 128.0);
          const double ref[3] = {luma + 2 * (1 - kb) * u, luma - 2 * kb * (1 - kb) / kg * u - 2 * kr * (1 - kr) / kg * v,
                                 luma + 2 * (1 - kr) * v};
          for (int k = 0; k < 3; ++k)
            maxError = std::max(maxError, std::abs(bgr.at<cv::Vec3b>(r, c)[k] - std::clamp(ref[k], 0.0, 255.0)));
          maxError = std::max(maxError, std::abs(gray.at<std::uint8_t>(r, c) - std::clamp(luma, 0.0, 255.0)));
        }
      EXPECT_LE(maxError, 0.5 + 1e-3) << "bt601 " << bt601 << " full " << full;
    }
}

TEST(GpuImage, AreaResizeMatchesOpenCv) {
  if (!haveDevice()) GTEST_SKIP() << "no CUDA device";
  // The camera sizes of the pipeline (2896 -> 1936 -> 968) scaled down, plus an odd factor and an integer one.
  const int w = 362, h = 236;
  const cv::Mat y = randomPlane(w, h);
  cv::Mat uv(h / 2, w / 2, CV_8UC2);
  cv::randu(uv, 0, 256);
  for (const int channels : {1, 3}) {
    const cv::Mat full = gpuConvert(y, uv, {false, true}, channels, {});
    for (const std::vector<double>& scales :
         {std::vector<double>{1936.0 / 2896.0, 0.5}, {0.5}, {1.0 / 3.0}, {0.37}, {0.8, 0.5}}) {
      cv::Mat ref = full.clone();
      for (const double s : scales) cv::resize(ref, ref, {}, s, s, cv::INTER_AREA);
      const cv::Mat out = gpuConvert(y, uv, {false, true}, channels, scales);
      ASSERT_EQ(out.size(), ref.size());
      cv::Mat diff;
      cv::absdiff(out, ref, diff);
      double maxDiff;
      cv::minMaxLoc(diff.reshape(1), nullptr, &maxDiff);
      EXPECT_EQ(maxDiff, 0) << channels << " channels, first scale " << scales[0];
    }
  }
}

}  // namespace
