#pragma once

#include <cmath>

#include <Eigen/Core>

namespace sdv {

// EUCM: Khomutenko et al., RA-L 2016 (eq. 11 projection, 50 Jacobian, 37 inverse).
// Projection domain z > -w*d from Usenko et al., Double Sphere, 3DV 2018.
// Integer pixel coordinates are pixel centres. A small negative alpha (outside the EUCM paper range) models
// residual pincushion distortion of rectified images, e.g. KITTI (see docs/ROADMAP.md, step 11).
class Camera {
 public:
  double fx = 0, fy = 0, cx = 0, cy = 0;
  double alpha = 0, beta = 1;
  int width = 0, height = 0;

  static Camera pinhole(double fx, double fy, double cx, double cy, int w, int h) {
    return Camera{fx, fy, cx, cy, 0.0, 1.0, w, h};
  }
  static Camera eucm(double fx, double fy, double cx, double cy, double alpha, double beta, int w,
                     int h) {
    return Camera{fx, fy, cx, cy, alpha, beta, w, h};
  }

  bool isPinhole() const { return alpha == 0.0 && beta == 1.0; }

  // Coarse pixel j covers fine pixels 2j, 2j+1: u_coarse = (u_fine - 0.5) / 2.
  Camera atLevel(int level) const {
    Camera c = *this;
    for (int i = 0; i < level; ++i) {
      c.fx *= 0.5;
      c.fy *= 0.5;
      c.cx = (c.cx + 0.5) * 0.5 - 0.5;
      c.cy = (c.cy + 0.5) * 0.5 - 0.5;
      c.width /= 2;
      c.height /= 2;
    }
    return c;
  }

  template <typename T>
  bool project(const Eigen::Matrix<T, 3, 1>& p, Eigen::Matrix<T, 2, 1>& uv) const {
    const T d = std::sqrt(T(beta) * (p.x() * p.x() + p.y() * p.y()) + p.z() * p.z());
    const T den = T(alpha) * d + (T(1) - T(alpha)) * p.z();
    if (!(p.z() > -T(validityW()) * d) || den <= T(1e-9)) return false;
    uv.x() = T(fx) * p.x() / den + T(cx);
    uv.y() = T(fy) * p.y() / den + T(cy);
    return true;
  }

  template <typename T>
  bool project(const Eigen::Matrix<T, 3, 1>& p, Eigen::Matrix<T, 2, 1>& uv,
               Eigen::Matrix<T, 2, 3>& J) const {
    const T x = p.x(), y = p.y(), z = p.z();
    const T d = std::sqrt(T(beta) * (x * x + y * y) + z * z);
    const T den = T(alpha) * d + (T(1) - T(alpha)) * z;
    if (!(z > -T(validityW()) * d) || den <= T(1e-9)) return false;
    const T invDen = T(1) / den;
    uv.x() = T(fx) * x * invDen + T(cx);
    uv.y() = T(fy) * y * invDen + T(cy);

    const T invD = T(1) / d;
    const T dDx = T(alpha * beta) * x * invD;
    const T dDy = T(alpha * beta) * y * invD;
    const T dDz = T(alpha) * z * invD + (T(1) - T(alpha));
    const T invDen2 = invDen * invDen;
    J(0, 0) = T(fx) * (den - x * dDx) * invDen2;
    J(0, 1) = -T(fx) * x * dDy * invDen2;
    J(0, 2) = -T(fx) * x * dDz * invDen2;
    J(1, 0) = -T(fy) * y * dDx * invDen2;
    J(1, 1) = T(fy) * (den - y * dDy) * invDen2;
    J(1, 2) = -T(fy) * y * dDz * invDen2;
    return true;
  }

  // Returns a unit-norm bearing.
  template <typename T>
  bool unproject(const Eigen::Matrix<T, 2, 1>& uv, Eigen::Matrix<T, 3, 1>& bearing) const {
    const T mx = (uv.x() - T(cx)) / T(fx);
    const T my = (uv.y() - T(cy)) / T(fy);
    const T r2 = mx * mx + my * my;
    const T k = T(beta) * (T(2) * T(alpha) - T(1)) * r2;
    if (alpha > 0.5 && k >= T(1)) return false;
    const T mz = (T(1) - T(beta * alpha * alpha) * r2) /
                 (T(alpha) * std::sqrt(T(1) - k) + (T(1) - T(alpha)));
    bearing = Eigen::Matrix<T, 3, 1>(mx, my, mz).normalized();
    return true;
  }

  bool isInside(double u, double v, double border) const {
    return u >= border && v >= border && u < width - 1 - border && v < height - 1 - border;
  }

 private:
  double validityW() const {
    return alpha <= 0.5 ? alpha / (1.0 - alpha) : (1.0 - alpha) / alpha;
  }
};

}  // namespace sdv
