#include <sdv/brightness_fit.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <execution>
#include <numeric>
#include <stdexcept>

#include <sdv/image_pyramid.h>

namespace sdv {

namespace {

struct Sample {
  double irradiance, intensity, gradientWeight;
};

// intensity = alpha * irradiance + beta, weighted and Huber-reweighted.
bool fitLine(const std::vector<Sample>& samples, const BrightnessFitSettings& s, double& alpha, double& beta,
             int& inliers) {
  alpha = 1.0;
  beta = 0.0;
  for (int it = 0; it < s.iterations; ++it) {
    double sw = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    for (const Sample& p : samples) {
      const double r = std::abs(p.intensity - alpha * p.irradiance - beta);
      // The first pass has no model yet: gradient weights only. Huber alone still lets gross outliers
      // (occlusions) pull the line, so the second half of the passes drops them.
      double w = it == 0 || r <= s.huberThreshold ? 1.0 : s.huberThreshold / r;
      if (2 * it >= s.iterations && r > 3 * s.huberThreshold) w = 0;
      w *= p.gradientWeight;
      sw += w, sx += w * p.irradiance, sy += w * p.intensity;
      sxx += w * p.irradiance * p.irradiance, syy += w * p.intensity * p.intensity;
      sxy += w * p.irradiance * p.intensity;
    }
    if (sw <= 0) return false;
    // Orthogonal regression: the irradiance comes from an image as noisy as the target, and ordinary least squares
    // would flatten the slope by var(J) / (var(J) + noise).
    const double mx = sx / sw, my = sy / sw;
    const double vxx = sxx / sw - mx * mx, vyy = syy / sw - my * my, vxy = sxy / sw - mx * my;
    if (vxy <= 1e-9 * (vxx + vyy)) return false;
    alpha = (vyy - vxx + std::sqrt((vyy - vxx) * (vyy - vxx) + 4 * vxy * vxy)) / (2 * vxy);
    beta = my - alpha * mx;
  }
  inliers = 0;
  for (const Sample& p : samples) inliers += std::abs(p.intensity - alpha * p.irradiance - beta) <= s.huberThreshold;
  return alpha > 0 && inliers >= s.minPoints;
}

}  // namespace

BrightnessFitResult fitCameraBrightness(const Camera& cam, const std::vector<Sophus::SE3d>& T_c_w,
                                        const std::vector<cv::Mat>& images, const std::vector<IrradiancePoint>& points,
                                        const BrightnessFitSettings& settings) {
  if (images.size() != T_c_w.size()) throw std::invalid_argument("one image per keyframe pose required");
  const int n = static_cast<int>(T_c_w.size());
  BrightnessFitResult out{std::vector<AffineBrightness>(n), std::vector<int>(n, 0)};
  std::vector<char> fitted(n, 0);
  const double c2 = settings.gradientWeightC * settings.gradientWeightC;
  std::vector<int> keyframes(n);
  std::iota(keyframes.begin(), keyframes.end(), 0);
  std::for_each(std::execution::par, keyframes.begin(), keyframes.end(), [&](int k) {
    const ImagePyramid pyr(toFloatGray(images[k]), 1);
    const ImageLevel& img = pyr.level(0);
    std::vector<Sample> samples;
    for (const IrradiancePoint& p : points) {
      if (std::abs(p.keyframe - k) > settings.hostWindow) continue;
      Eigen::Vector2d uv;
      const Eigen::Vector3d x = T_c_w[k] * p.position;
      if (!cam.project(x, uv) || !cam.isInside(uv.x(), uv.y(), 2.0)) continue;
      const Eigen::Vector3f v = img.interpolate(static_cast<float>(uv.x()), static_cast<float>(uv.y()));
      samples.push_back({p.irradiance, v[0], c2 / (c2 + v[1] * v[1] + v[2] * v[2])});
    }
    double alpha, beta;
    int inliers = 0;
    if (static_cast<int>(samples.size()) < settings.minPoints || !fitLine(samples, settings, alpha, beta, inliers))
      return;
    out.affine[k] = {std::log(alpha), beta};
    out.inliers[k] = inliers;
    fitted[k] = 1;
  });
  for (int k = 0; k < n; ++k) {
    if (fitted[k]) continue;
    for (int d = 1; d < n; ++d) {
      if (k - d >= 0 && fitted[k - d]) {
        out.affine[k] = out.affine[k - d];
        break;
      }
      if (k + d < n && fitted[k + d]) {
        out.affine[k] = out.affine[k + d];
        break;
      }
    }
  }
  return out;
}

}  // namespace sdv
