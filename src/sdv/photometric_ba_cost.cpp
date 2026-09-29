#include <sdv/photometric_ba_cost.h>

#include <cmath>

namespace sdv::pba {

namespace {

Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
  Eigen::Matrix3d m;
  m << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
  return m;
}

// Pixel k of the pattern: residual and dr/dx (x = R b + rho t, the point in the target camera scaled by rho).
bool pixel(const PointData& p, const Camera& cam, const Interpolator& image, int k, const Eigen::Vector3d& x,
           double scale, const double* ah, const double* at, double& residual, Eigen::RowVector3d* dRdX) {
  Eigen::Vector2d uv;
  Eigen::Matrix<double, 2, 3> dUvdX;
  if (!cam.project(x, uv, dUvdX)) return false;
  double intensity, dIdRow, dIdCol;
  image.Evaluate(uv.y(), uv.x(), &intensity, &dIdRow, &dIdCol);
  const double s = std::sqrt(p.pattern.gradientWeights[k]);
  residual = s * ((intensity - at[1]) - scale * (p.pattern.intensities[k] - ah[1]));
  if (dRdX) *dRdX = s * Eigen::RowVector2d(dIdCol, dIdRow) * dUvdX;
  return true;
}

void affineJacobians(const PointData& p, int k, double scale, const double* ah, double** jacobians, int hostIndex,
                     int targetIndex) {
  const double s = std::sqrt(p.pattern.gradientWeights[k]);
  const double hostTerm = p.pattern.intensities[k] - ah[1];
  if (jacobians[hostIndex]) {
    jacobians[hostIndex][2 * k] = s * scale * hostTerm;
    jacobians[hostIndex][2 * k + 1] = s * scale;
  }
  if (jacobians[targetIndex]) {
    jacobians[targetIndex][2 * k] = -s * scale * hostTerm;
    jacobians[targetIndex][2 * k + 1] = -s;
  }
}

template <int Rows, int Cols>
void zero(double** jacobians, int index) {
  if (jacobians && jacobians[index]) std::fill_n(jacobians[index], Rows * Cols, 0.0);
}

// Point Jacobians (3 x 6) of x = R b + rho t for T_t_h exp(xi) (host side) and exp(-xi) T_t_h (target side).
Eigen::Matrix<double, 3, 6> hostSide(const Eigen::Matrix3d& R, const Eigen::Vector3d& b, double rho) {
  Eigen::Matrix<double, 3, 6> dx;
  dx << rho * R, -R * skew(b);
  return dx;
}

Eigen::Matrix<double, 3, 6> targetSide(const Eigen::Vector3d& x, double rho) {
  Eigen::Matrix<double, 3, 6> dx;
  dx << -rho * Eigen::Matrix3d::Identity(), skew(x);
  return dx;
}

// Pattern residuals for T_t_h; `fill(k, g, x, b)` writes the pose Jacobians of pixel k (g = dr/dx). Inverse depth
// and affine Jacobians go to blocks rhoIndex, rhoIndex + 1, rhoIndex + 2; all blocks are zeroed first.
template <typename Fill>
void evaluatePattern(const PointData& p, const Camera& cam, const Interpolator& image, const Sophus::SE3d& T_t_h,
                     const double* const* parameters, int rhoIndex, int blocks, const int* blockSizes,
                     double* residuals, double** jacobians, Fill fill) {
  const double rho = parameters[rhoIndex][0];
  const double *ah = parameters[rhoIndex + 1], *at = parameters[rhoIndex + 2];
  const Eigen::Matrix3d R = T_t_h.rotationMatrix();
  const Eigen::Vector3d& t = T_t_h.translation();
  const double scale = std::exp(at[0] - ah[0]);
  if (jacobians)
    for (int i = 0; i < blocks; ++i)
      if (jacobians[i]) std::fill_n(jacobians[i], kPatternSize * blockSizes[i], 0.0);
  for (int k = 0; k < kPatternSize; ++k) {
    const Eigen::Vector3d& b = p.pattern.bearings[k];
    const Eigen::Vector3d x = R * b + rho * t;
    Eigen::RowVector3d g;
    if (!pixel(p, cam, image, k, x, scale, ah, at, residuals[k], jacobians ? &g : nullptr)) {
      residuals[k] = 0;
      continue;
    }
    if (!jacobians) continue;
    fill(k, g, R, x, b, rho);
    if (jacobians[rhoIndex]) jacobians[rhoIndex][k] = g.dot(t);
    affineJacobians(p, k, scale, ah, jacobians, rhoIndex + 1, rhoIndex + 2);
  }
}

void setRow(double** jacobians, int index, int k, const Eigen::Matrix<double, 1, 6>& row) {
  if (jacobians[index]) Eigen::Map<Eigen::Matrix<double, 1, 6>>(jacobians[index] + 7 * k) = row;
}

}  // namespace

bool SE3TangentManifold::Plus(const double* x, const double* delta, double* x_plus_delta) const {
  Eigen::Map<Sophus::SE3d> out(x_plus_delta);
  out =
      Eigen::Map<const Sophus::SE3d>(x) * Sophus::SE3d::exp(Eigen::Map<const Sophus::Vector6d>(delta));
  return true;
}

bool SE3TangentManifold::PlusJacobian(const double*, double* jacobian) const {
  Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> J(jacobian);
  J.setZero();
  J.topRows<6>().setIdentity();
  return true;
}

bool SE3TangentManifold::Minus(const double* y, const double* x, double* y_minus_x) const {
  Eigen::Map<Sophus::Vector6d> out(y_minus_x);
  out =
      (Eigen::Map<const Sophus::SE3d>(x).inverse() * Eigen::Map<const Sophus::SE3d>(y)).log();
  return true;
}

bool SE3TangentManifold::MinusJacobian(const double*, double* jacobian) const {
  Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J(jacobian);
  J.setZero();
  J.leftCols<6>().setIdentity();
  return true;
}

TemporalCost::TemporalCost(const PointData* point, const Camera* cam, const Interpolator* image,
                           const Sophus::SE3d& T_ct_b, const Sophus::SE3d& T_b_ch)
    : m_point(point), m_cam(cam), m_image(image), m_T_ct_b(T_ct_b), m_T_b_ch(T_b_ch),
      m_adjHost(T_b_ch.inverse().Adj()), m_adjTarget(T_ct_b.Adj()) {}

bool TemporalCost::Evaluate(const double* const* parameters, double* residuals, double** jacobians) const {
  const Eigen::Map<const Sophus::SE3d> T_w_h(parameters[0]), T_w_t(parameters[1]);
  const double rho = parameters[2][0];
  const double *ah = parameters[3], *at = parameters[4];
  const Sophus::SE3d T_t_h = m_T_ct_b * T_w_t.inverse() * T_w_h * m_T_b_ch;
  const Eigen::Matrix3d R = T_t_h.rotationMatrix();
  const Eigen::Vector3d& t = T_t_h.translation();
  const double scale = std::exp(at[0] - ah[0]);
  zero<kPatternSize, 7>(jacobians, 0);
  zero<kPatternSize, 7>(jacobians, 1);
  zero<kPatternSize, 1>(jacobians, 2);
  zero<kPatternSize, 2>(jacobians, 3);
  zero<kPatternSize, 2>(jacobians, 4);
  for (int k = 0; k < kPatternSize; ++k) {
    const Eigen::Vector3d& b = m_point->pattern.bearings[k];
    const Eigen::Vector3d x = R * b + rho * t;
    Eigen::RowVector3d g;
    if (!pixel(*m_point, *m_cam, *m_image, k, x, scale, ah, at, residuals[k], jacobians ? &g : nullptr)) {
      residuals[k] = 0;
      continue;
    }
    if (!jacobians) continue;
    // Host: T_t_h exp(Adj(T_b_ch^-1) xi); target: exp(-Adj(T_ct_b) xi) T_t_h (see the Jacobian test).
    if (jacobians[0]) {
      Eigen::Matrix<double, 3, 6> dx;
      dx << rho * R, -R * skew(b);
      Eigen::Map<Eigen::Matrix<double, 1, 6>>(jacobians[0] + 7 * k) = g * dx * m_adjHost;
    }
    if (jacobians[1]) {
      Eigen::Matrix<double, 3, 6> dx;
      dx << -rho * Eigen::Matrix3d::Identity(), skew(x);
      Eigen::Map<Eigen::Matrix<double, 1, 6>>(jacobians[1] + 7 * k) = g * dx * m_adjTarget;
    }
    if (jacobians[2]) jacobians[2][k] = g.dot(t);
    affineJacobians(*m_point, k, scale, ah, jacobians, 3, 4);
  }
  return true;
}

bool StaticCost::Evaluate(const double* const* parameters, double* residuals, double** jacobians) const {
  const double rho = parameters[0][0];
  const double *ah = parameters[1], *at = parameters[2];
  const Eigen::Matrix3d R = m_T_t_h.rotationMatrix();
  const Eigen::Vector3d& t = m_T_t_h.translation();
  const double scale = std::exp(at[0] - ah[0]);
  zero<kPatternSize, 1>(jacobians, 0);
  zero<kPatternSize, 2>(jacobians, 1);
  zero<kPatternSize, 2>(jacobians, 2);
  for (int k = 0; k < kPatternSize; ++k) {
    const Eigen::Vector3d x = R * m_point->pattern.bearings[k] + rho * t;
    Eigen::RowVector3d g;
    if (!pixel(*m_point, *m_cam, *m_image, k, x, scale, ah, at, residuals[k], jacobians ? &g : nullptr)) {
      residuals[k] = 0;
      continue;
    }
    if (!jacobians) continue;
    if (jacobians[0]) jacobians[0][k] = g.dot(t);
    affineJacobians(*m_point, k, scale, ah, jacobians, 1, 2);
  }
  return true;
}

bool TemporalExtrinsicCost::Evaluate(const double* const* parameters, double* residuals, double** jacobians) const {
  const Eigen::Map<const Sophus::SE3d> T_w_h(parameters[0]), T_w_t(parameters[1]);
  const Eigen::Map<const Sophus::SE3d> T_b_ch(parameters[2]), T_b_ct(parameters[3]);
  const Sophus::SE3d T_ct_b = T_b_ct.inverse();
  const Sophus::SE3d T_t_h = T_ct_b * T_w_t.inverse() * T_w_h * T_b_ch;
  // Poses as in TemporalCost; extrinsics act directly on T_t_h (no adjoint): host T_t_h exp(xi), target
  // exp(-xi) T_t_h.
  const Eigen::Matrix<double, 6, 6> adjHost = T_b_ch.inverse().Adj(), adjTarget = T_ct_b.Adj();
  static constexpr int sizes[] = {7, 7, 7, 7, 1, 2, 2};
  evaluatePattern(*m_point, *m_cam, *m_image, T_t_h, parameters, 4, 7, sizes, residuals, jacobians,
                  [&](int k, const Eigen::RowVector3d& g, const Eigen::Matrix3d& R, const Eigen::Vector3d& x,
                      const Eigen::Vector3d& b, double rho) {
                    const Eigen::Matrix<double, 1, 6> gh = g * hostSide(R, b, rho), gt = g * targetSide(x, rho);
                    setRow(jacobians, 0, k, gh * adjHost);
                    setRow(jacobians, 1, k, gt * adjTarget);
                    setRow(jacobians, 2, k, gh);
                    setRow(jacobians, 3, k, gt);
                  });
  return true;
}

bool TemporalSameCameraExtrinsicCost::Evaluate(const double* const* parameters, double* residuals,
                                               double** jacobians) const {
  const Eigen::Map<const Sophus::SE3d> T_w_h(parameters[0]), T_w_t(parameters[1]), T_b_c(parameters[2]);
  const Sophus::SE3d T_c_b = T_b_c.inverse();
  const Sophus::SE3d T_t_h = T_c_b * T_w_t.inverse() * T_w_h * T_b_c;
  const Eigen::Matrix<double, 6, 6> adjHost = T_c_b.Adj(), adjTarget = T_c_b.Adj();
  static constexpr int sizes[] = {7, 7, 7, 1, 2, 2};
  // T_b_c exp(xi) enters on both sides: exp(-xi) T_t_h exp(xi).
  evaluatePattern(*m_point, *m_cam, *m_image, T_t_h, parameters, 3, 6, sizes, residuals, jacobians,
                  [&](int k, const Eigen::RowVector3d& g, const Eigen::Matrix3d& R, const Eigen::Vector3d& x,
                      const Eigen::Vector3d& b, double rho) {
                    const Eigen::Matrix<double, 1, 6> gh = g * hostSide(R, b, rho), gt = g * targetSide(x, rho);
                    setRow(jacobians, 0, k, gh * adjHost);
                    setRow(jacobians, 1, k, gt * adjTarget);
                    setRow(jacobians, 2, k, gh + gt);
                  });
  return true;
}

bool StaticExtrinsicCost::Evaluate(const double* const* parameters, double* residuals, double** jacobians) const {
  const Eigen::Map<const Sophus::SE3d> T_b_ch(parameters[0]), T_b_ct(parameters[1]);
  const Sophus::SE3d T_t_h = T_b_ct.inverse() * T_b_ch;
  static constexpr int sizes[] = {7, 7, 1, 2, 2};
  evaluatePattern(*m_point, *m_cam, *m_image, T_t_h, parameters, 2, 5, sizes, residuals, jacobians,
                  [&](int k, const Eigen::RowVector3d& g, const Eigen::Matrix3d& R, const Eigen::Vector3d& x,
                      const Eigen::Vector3d& b, double rho) {
                    setRow(jacobians, 0, k, g * hostSide(R, b, rho));
                    setRow(jacobians, 1, k, g * targetSide(x, rho));
                  });
  return true;
}

RelativePoseCost::RelativePoseCost(const Sophus::SE3d& T_a_b, double sigmaT, double sigmaRDeg)
    : m_T_b_a_measured(T_a_b.inverse()) {
  const double r = sigmaRDeg * M_PI / 180.0;
  m_weight << 1 / sigmaT, 1 / sigmaT, 1 / sigmaT, 1 / r, 1 / r, 1 / r;
}

bool RelativePoseCost::Evaluate(const double* const* parameters, double* residuals, double** jacobians) const {
  const Eigen::Map<const Sophus::SE3d> T_w_a(parameters[0]), T_w_b(parameters[1]);
  const Sophus::SE3d T_a_b = T_w_a.inverse() * T_w_b;
  Eigen::Map<Sophus::Vector6d> r(residuals);
  r = m_weight.asDiagonal() * (m_T_b_a_measured * T_a_b).log();
  if (!jacobians) return true;
  // e = log(M T_a_b): T_w_b exp(xi) adds ~xi, T_w_a exp(xi) gives exp(-xi) in front of T_a_b, i.e. -Adj(T_b_a) xi.
  if (jacobians[0]) {
    Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J(jacobians[0]);
    J.setZero();
    J.leftCols<6>() = -(m_weight.asDiagonal() * T_a_b.inverse().Adj());
  }
  if (jacobians[1]) {
    Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J(jacobians[1]);
    J.setZero();
    J.leftCols<6>() = m_weight.asDiagonal();
  }
  return true;
}

void patternResidual(const PointData& p, const Camera& cam, const Interpolator& image, const Sophus::SE3d& T_t_h,
                     double rho, const double* ah, const double* at, double* residual) {
  const double scale = std::exp(at[0] - ah[0]);
  for (int k = 0; k < kPatternSize; ++k) {
    const Eigen::Vector3d x = T_t_h.so3() * p.pattern.bearings[k] + rho * T_t_h.translation();
    if (!pixel(p, cam, image, k, x, scale, ah, at, residual[k], nullptr)) residual[k] = 0;
  }
}

}  // namespace sdv::pba
