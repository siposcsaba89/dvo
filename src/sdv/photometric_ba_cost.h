#pragma once

#include <cstdint>

#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <Eigen/Core>
#include <sophus/se3.hpp>

#include <sdv/camera.h>
#include <sdv/photometric.h>

// Cost functions of the photometric bundle adjustment with analytic Jacobians (exposed for the Jacobian tests).
namespace sdv::pba {

using Grid = ceres::Grid2D<std::uint8_t, 1>;  // 8-bit images: a quarter of the memory of float grids
using Interpolator = ceres::BiCubicInterpolator<Grid>;

struct PointData {
  int host, hostCam;
  PatternPoint pattern;
  Eigen::Vector3d bearing;  // centre
};

// SE3 poses T_w_b in Sophus layout with right increments T * exp(delta). PlusJacobian is [I; 0], so the cost
// functions below write the Jacobian with respect to delta into the first six columns of their 7-column blocks.
class SE3TangentManifold : public ceres::Manifold {
 public:
  int AmbientSize() const override { return 7; }
  int TangentSize() const override { return 6; }
  bool Plus(const double* x, const double* delta, double* x_plus_delta) const override;
  bool PlusJacobian(const double* x, double* jacobian) const override;
  bool Minus(const double* y, const double* x, double* y_minus_x) const override;
  bool MinusJacobian(const double* x, double* jacobian) const override;
};

// Pattern residual of a point in a camera of another keyframe, relative pose T_t_h = T_ct_b T_w_t^-1 T_w_h T_b_ch,
// affine brightness I_t - b_t = exp(a_t - a_h) (I_h - b_h), each pixel weighted by the square root of its gradient
// weight. Parameters: host T_w_b (7), target T_w_b (7), inverse depth (1), host affine (2), target affine (2).
class TemporalCost : public ceres::SizedCostFunction<kPatternSize, 7, 7, 1, 2, 2> {
 public:
  TemporalCost(const PointData* point, const Camera* cam, const Interpolator* image, const Sophus::SE3d& T_ct_b,
               const Sophus::SE3d& T_b_ch);
  bool Evaluate(const double* const* parameters, double* residuals, double** jacobians) const override;

 private:
  const PointData* m_point;
  const Camera* m_cam;
  const Interpolator* m_image;
  Sophus::SE3d m_T_ct_b, m_T_b_ch;
  Eigen::Matrix<double, 6, 6> m_adjHost, m_adjTarget;
};

// Another camera of the host keyframe (fixed extrinsic T_t_h). Parameters: inverse depth, host affine, target affine.
class StaticCost : public ceres::SizedCostFunction<kPatternSize, 1, 2, 2> {
 public:
  StaticCost(const PointData* point, const Camera* cam, const Interpolator* image, const Sophus::SE3d& T_t_h)
      : m_point(point), m_cam(cam), m_image(image), m_T_t_h(T_t_h) {}
  bool Evaluate(const double* const* parameters, double* residuals, double** jacobians) const override;

 private:
  const PointData* m_point;
  const Camera* m_cam;
  const Interpolator* m_image;
  Sophus::SE3d m_T_t_h;
};

// Costs with free rig extrinsics: T_b_c blocks (7, SE3TangentManifold). T_b_c exp(delta) is the left perturbation
// T_c_b <- exp(-delta) T_c_b of the camera-from-body transform, so the extrinsics share the manifold of the poses.
// Temporal residual between two cameras. Parameters: host T_w_b, target T_w_b, host T_b_c, target T_b_c, inverse
// depth, host affine, target affine.
class TemporalExtrinsicCost : public ceres::SizedCostFunction<kPatternSize, 7, 7, 7, 7, 1, 2, 2> {
 public:
  TemporalExtrinsicCost(const PointData* point, const Camera* cam, const Interpolator* image)
      : m_point(point), m_cam(cam), m_image(image) {}
  bool Evaluate(const double* const* parameters, double* residuals, double** jacobians) const override;

 private:
  const PointData* m_point;
  const Camera* m_cam;
  const Interpolator* m_image;
};

// Temporal residual in the host camera itself (one extrinsic block for host and target). Parameters: host T_w_b,
// target T_w_b, T_b_c, inverse depth, host affine, target affine.
class TemporalSameCameraExtrinsicCost : public ceres::SizedCostFunction<kPatternSize, 7, 7, 7, 1, 2, 2> {
 public:
  TemporalSameCameraExtrinsicCost(const PointData* point, const Camera* cam, const Interpolator* image)
      : m_point(point), m_cam(cam), m_image(image) {}
  bool Evaluate(const double* const* parameters, double* residuals, double** jacobians) const override;

 private:
  const PointData* m_point;
  const Camera* m_cam;
  const Interpolator* m_image;
};

// Another camera of the host keyframe, T_t_h = T_ct_b T_b_ch. Parameters: host T_b_c, target T_b_c, inverse depth,
// host affine, target affine.
class StaticExtrinsicCost : public ceres::SizedCostFunction<kPatternSize, 7, 7, 1, 2, 2> {
 public:
  StaticExtrinsicCost(const PointData* point, const Camera* cam, const Interpolator* image)
      : m_point(point), m_cam(cam), m_image(image) {}
  bool Evaluate(const double* const* parameters, double* residuals, double** jacobians) const override;

 private:
  const PointData* m_point;
  const Camera* m_cam;
  const Interpolator* m_image;
};

// Relative pose prior log(T_a_b_measured^-1 T_w_a^-1 T_w_b) weighted by the sigmas, for SE3TangentManifold poses;
// Jacobians with the right Jacobian of SE3 taken as identity (small residuals).
class RelativePoseCost : public ceres::SizedCostFunction<6, 7, 7> {
 public:
  RelativePoseCost(const Sophus::SE3d& T_a_b, double sigmaT, double sigmaRDeg);
  bool Evaluate(const double* const* parameters, double* residuals, double** jacobians) const override;

 private:
  Sophus::SE3d m_T_b_a_measured;
  Sophus::Vector6d m_weight;
};

// Residuals only (initial checks), same model as the costs.
void patternResidual(const PointData& p, const Camera& cam, const Interpolator& image, const Sophus::SE3d& T_t_h,
                     double rho, const double* ah, const double* at, double* residual);

}  // namespace sdv::pba
