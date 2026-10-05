#include <sdv/aimrec/gpu_image.h>

#include <cfloat>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace sdv::aimrec::gpu {

namespace {

void check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

template <typename T>
struct DeviceBuffer {
  T* data = nullptr;
  std::size_t size = 0;
  ~DeviceBuffer() { cudaFree(data); }
  void reserve(std::size_t n) {
    if (n <= size) return;
    cudaFree(data);
    data = nullptr;
    size = 0;
    check(cudaMalloc(&data, n * sizeof(T)), "cudaMalloc");
    size = n;
  }
};

struct AreaTap {
  int index;
  float weight;
};

// Taps of every destination pixel along one axis, in the order cv::resize(INTER_AREA) accumulates them: the source
// cell [d * scale, (d + 1) * scale) with fractional coverage at both ends, normalised by the cell width.
void areaTaps(int srcSize, int dstSize, double scale, std::vector<int>& begin, std::vector<AreaTap>& taps) {
  begin.assign(1, 0);
  taps.clear();
  for (int d = 0; d < dstSize; ++d) {
    const double f1 = d * scale, f2 = f1 + scale;
    const double cell = std::min(scale, srcSize - f1);
    int s1 = static_cast<int>(std::ceil(f1)), s2 = static_cast<int>(std::floor(f2));
    s2 = std::min(s2, srcSize - 1);
    s1 = std::min(s1, s2);
    if (s1 - f1 > 1e-3) taps.push_back({s1 - 1, static_cast<float>((s1 - f1) / cell)});
    for (int s = s1; s < s2; ++s) taps.push_back({s, static_cast<float>(1.0 / cell)});
    if (f2 - s2 > 1e-3) taps.push_back({s2, static_cast<float>(std::min(std::min(f2 - s2, 1.0), cell) / cell)});
    begin.push_back(static_cast<int>(taps.size()));
  }
}

__device__ unsigned char saturateRound(float v) {
  return static_cast<unsigned char>(min(max(__float2int_rn(v), 0), 255));
}

struct Yuv {
  float yScale, yOffset, cScale, rv, gu, gv, bu;
};

Yuv yuvCoefficients(ColorMatrix color) {
  const double kr = color.bt601 ? 0.299 : 0.2126, kb = color.bt601 ? 0.114 : 0.0722, kg = 1 - kr - kb;
  Yuv c;
  c.yScale = color.fullRange ? 1.0f : static_cast<float>(255.0 / 219.0);
  c.yOffset = color.fullRange ? 0.0f : 16.0f;
  c.cScale = color.fullRange ? 1.0f : static_cast<float>(255.0 / 224.0);
  c.rv = static_cast<float>(2 * (1 - kr));
  c.bu = static_cast<float>(2 * (1 - kb));
  c.gu = static_cast<float>(2 * kb * (1 - kb) / kg);
  c.gv = static_cast<float>(2 * kr * (1 - kr) / kg);
  return c;
}

// Chroma is taken from the 2x2 block the pixel belongs to, as the CPU colour conversion of 4:2:0 frames does.
__global__ void nv12ToImage(const unsigned char* y, std::size_t yPitch, const unsigned char* uv, std::size_t uvPitch,
                            int width, int height, Yuv k, int channels, unsigned char* dst, std::size_t dstPitch) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || row >= height) return;
  const float luma = k.yScale * (y[row * yPitch + x] - k.yOffset);
  unsigned char* out = dst + row * dstPitch + x * channels;
  if (channels == 1) {
    out[0] = saturateRound(luma);
    return;
  }
  const unsigned char* c = uv + (row / 2) * uvPitch + (x / 2) * 2;
  const float u = k.cScale * (c[0] - 128.0f), v = k.cScale * (c[1] - 128.0f);
  out[0] = saturateRound(luma + k.bu * u);
  out[1] = saturateRound(luma - k.gu * u - k.gv * v);
  out[2] = saturateRound(luma + k.rv * v);
}

// No fused multiply-adds: the sums are rounded like the CPU's separate multiplies and adds.
__global__ void areaResize(const unsigned char* src, std::size_t srcPitch, int channels, const int* xBegin,
                           const AreaTap* xTaps, const int* yBegin, const AreaTap* yTaps, int dstWidth, int dstHeight,
                           unsigned char* dst, std::size_t dstPitch) {
  const int dx = blockIdx.x * blockDim.x + threadIdx.x, dy = blockIdx.y * blockDim.y + threadIdx.y;
  if (dx >= dstWidth || dy >= dstHeight) return;
  float sum[3] = {0, 0, 0};
  for (int j = yBegin[dy]; j < yBegin[dy + 1]; ++j) {
    const unsigned char* row = src + yTaps[j].index * srcPitch;
    float h[3] = {0, 0, 0};
    for (int i = xBegin[dx]; i < xBegin[dx + 1]; ++i) {
      const unsigned char* p = row + xTaps[i].index * channels;
      for (int c = 0; c < channels; ++c) h[c] = __fadd_rn(h[c], __fmul_rn(xTaps[i].weight, p[c]));
    }
    for (int c = 0; c < channels; ++c) sum[c] = __fadd_rn(sum[c], __fmul_rn(yTaps[j].weight, h[c]));
  }
  unsigned char* out = dst + dy * dstPitch + dx * channels;
  for (int c = 0; c < channels; ++c) out[c] = saturateRound(sum[c]);
}

// Integer factor n, as the CPU: full cells take the integer block sum, rounded half up for n = 2 and to nearest
// otherwise; cells cut by the image border average the pixels they have.
__global__ void areaResizeInteger(const unsigned char* src, std::size_t srcPitch, int srcWidth, int srcHeight,
                                  int channels, int n, int dstWidth, int dstHeight, unsigned char* dst,
                                  std::size_t dstPitch) {
  const int dx = blockIdx.x * blockDim.x + threadIdx.x, dy = blockIdx.y * blockDim.y + threadIdx.y;
  if (dx >= dstWidth || dy >= dstHeight) return;
  unsigned char* out = dst + dy * dstPitch + dx * channels;
  const int x0 = dx * n, y0 = dy * n;
  const bool full = x0 + n <= srcWidth && y0 + n <= srcHeight;
  const float inverseArea = 1.0f / static_cast<float>(n * n);
  for (int c = 0; c < channels; ++c) {
    int sum = 0, count = 0;
    for (int y = y0; y < min(y0 + n, srcHeight); ++y) {
      const unsigned char* row = src + y * srcPitch + c;
      for (int x = x0; x < min(x0 + n, srcWidth); ++x, ++count) sum += row[x * channels];
    }
    if (count == 0)
      out[c] = 0;
    else if (!full)
      out[c] = saturateRound(__fdiv_rn(static_cast<float>(sum), static_cast<float>(count)));
    else
      out[c] = n == 2 ? static_cast<unsigned char>((sum + 2) >> 2) : saturateRound(__fmul_rn(sum, inverseArea));
  }
}

dim3 grid(int width, int height, dim3 block) {
  return dim3((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
}

// As cv::resize: rounded half to even.
int scaledLength(int length, double scale) { return static_cast<int>(std::lrint(length * scale)); }

}  // namespace

void scaledSize(int width, int height, std::span<const double> scales, int& outWidth, int& outHeight) {
  for (const double s : scales) {
    if (s == 1.0) continue;
    width = scaledLength(width, s);
    height = scaledLength(height, s);
  }
  outWidth = width;
  outHeight = height;
}

struct Converter::Impl {
  struct Step {
    int srcWidth = 0, srcHeight = 0;
    double scale = 0;
    int dstWidth = 0, dstHeight = 0;
    int integerFactor = 0;  // > 0: integer downscale
    DeviceBuffer<int> xBegin, yBegin;
    DeviceBuffer<AreaTap> xTaps, yTaps;
    DeviceBuffer<unsigned char> output;
  };

  cudaStream_t stream = nullptr;
  DeviceBuffer<unsigned char> full;
  std::vector<std::unique_ptr<Step>> steps;

  Impl() { check(cudaStreamCreate(&stream), "cudaStreamCreate"); }
  ~Impl() {
    steps.clear();
    cudaStreamDestroy(stream);
  }

  Step& step(std::size_t i, int srcWidth, int srcHeight, double scale) {
    if (i >= steps.size()) steps.resize(i + 1);
    auto& s = steps[i];
    if (s && s->srcWidth == srcWidth && s->srcHeight == srcHeight && s->scale == scale) return *s;
    s = std::make_unique<Step>();
    s->srcWidth = srcWidth;
    s->srcHeight = srcHeight;
    s->scale = scale;
    s->dstWidth = scaledLength(srcWidth, scale);
    s->dstHeight = scaledLength(srcHeight, scale);
    // As cv::resize: the cell size is the inverse of the requested factor, not the ratio of the rounded sizes.
    const double cell = 1.0 / scale;
    const int n = static_cast<int>(std::lround(cell));
    if (std::abs(cell - n) < DBL_EPSILON) {
      s->integerFactor = n;
      return *s;
    }
    upload(srcWidth, s->dstWidth, cell, s->xBegin, s->xTaps);
    upload(srcHeight, s->dstHeight, cell, s->yBegin, s->yTaps);
    return *s;
  }

  void upload(int srcSize, int dstSize, double cell, DeviceBuffer<int>& begin, DeviceBuffer<AreaTap>& taps) {
    std::vector<int> b;
    std::vector<AreaTap> t;
    areaTaps(srcSize, dstSize, cell, b, t);
    begin.reserve(b.size());
    taps.reserve(t.size());
    check(cudaMemcpy(begin.data, b.data(), b.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy");
    check(cudaMemcpy(taps.data, t.data(), t.size() * sizeof(AreaTap), cudaMemcpyHostToDevice), "cudaMemcpy");
  }
};

Converter::Converter() : m_impl(std::make_unique<Impl>()) {}
Converter::~Converter() = default;

void Converter::convert(const Nv12Frame& src, ColorMatrix color, int channels, std::span<const double> scales,
                        std::uint8_t* dst, std::size_t dstStep) {
  if (channels != 1 && channels != 3) throw std::invalid_argument("gpu::Converter: 1 or 3 channels");
  Impl& m = *m_impl;
  const dim3 block(32, 8);
  const std::size_t fullPitch = static_cast<std::size_t>(src.width) * channels;
  m.full.reserve(fullPitch * src.height);
  nv12ToImage<<<grid(src.width, src.height, block), block, 0, m.stream>>>(
      src.y, src.yPitch, src.uv, src.uvPitch, src.width, src.height, yuvCoefficients(color), channels, m.full.data,
      fullPitch);
  check(cudaGetLastError(), "nv12ToImage");

  const unsigned char* image = m.full.data;
  std::size_t pitch = fullPitch;
  int width = src.width, height = src.height;
  std::size_t used = 0;
  for (const double scale : scales) {
    if (scale == 1.0) continue;
    Impl::Step& s = m.step(used++, width, height, scale);
    const std::size_t dstPitch = static_cast<std::size_t>(s.dstWidth) * channels;
    s.output.reserve(dstPitch * s.dstHeight);
    if (s.integerFactor > 0)
      areaResizeInteger<<<grid(s.dstWidth, s.dstHeight, block), block, 0, m.stream>>>(
          image, pitch, width, height, channels, s.integerFactor, s.dstWidth, s.dstHeight, s.output.data, dstPitch);
    else
      areaResize<<<grid(s.dstWidth, s.dstHeight, block), block, 0, m.stream>>>(
          image, pitch, channels, s.xBegin.data, s.xTaps.data, s.yBegin.data, s.yTaps.data, s.dstWidth, s.dstHeight,
          s.output.data, dstPitch);
    check(cudaGetLastError(), "areaResize");
    image = s.output.data;
    pitch = dstPitch;
    width = s.dstWidth;
    height = s.dstHeight;
  }
  check(cudaMemcpy2DAsync(dst, dstStep, image, pitch, static_cast<std::size_t>(width) * channels, height,
                          cudaMemcpyDeviceToHost, m.stream),
        "cudaMemcpy2DAsync");
  check(cudaStreamSynchronize(m.stream), "cudaStreamSynchronize");
}

}  // namespace sdv::aimrec::gpu
